#!/usr/bin/env python3
"""
Emergency Intersection Priority System - Standalone Version
AI that asks before it acts - Complete system in a single file
"""

import os
import sys
import json
import yaml
import argparse
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import seaborn as sns
    from io import BytesIO
    import base64
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages: pip install fastapi uvicorn pillow pydantic requests pyyaml matplotlib seaborn")
    sys.exit(1)

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

class VehicleType(Enum):
    """Vehicle types with their base priorities."""
    AMBULANCE = "ambulance"
    FIRE_ENGINE = "fire_engine"
    POLICE = "police"
    PRESIDENTIAL = "presidential"
    CIVILIAN = "civilian"

@dataclass
class Vehicle:
    """Internal representation of a detected vehicle."""
    id: str
    kind: VehicleType
    has_right_of_way_signal: bool
    distance_to_conflict_m: float
    arrival_time_s: float
    bearing_deg: float

class PolicyOutput(BaseModel):
    """Final output schema for the API."""
    ordered_ids: List[str]
    scores: Dict[str, float]
    reasoning: str
    legality_notes: str
    confidence: float = 1.0
    requires_human_confirmation: bool = False

class AnalysisRequest(BaseModel):
    """Request schema for analysis."""
    jurisdiction: str = "us"
    use_llm: bool = False

class RuleEngine:
    """Deterministic rule engine for priority calculation."""
    
    def __init__(self, jurisdiction: str = "us"):
        self.jurisdiction = jurisdiction
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load jurisdiction-specific configuration."""
        # Default US configuration
        default_config = {
            "traffic_side": "right",
            "base_priority": {
                "ambulance": 100,
                "fire_engine": 95,
                "police": 90,
                "presidential": 70,
                "civilian": 10
            },
            "bonuses": {
                "lights": 15,
                "proximity_weight": 0.2,
                "arrival_weight": 0.1
            },
            "tie_breaks": ["right_hand_rule", "apparatus_mass", "arrival_time"],
            "legal_notes": {
                "emergency": "General rule: yield to emergency vehicles with lights/sirens active.",
                "civilian": "Civilian vehicles follow normal traffic rules.",
                "presidential": "Presidential motorcades have special privileges but yield to emergency vehicles."
            }
        }
        
        # UK configuration override
        if self.jurisdiction.lower() == "uk":
            default_config.update({
                "traffic_side": "left",
                "tie_breaks": ["left_hand_rule", "apparatus_mass", "arrival_time"],
                "legal_notes": {
                    "emergency": "UK law: emergency vehicles with lights/sirens have priority.",
                    "civilian": "Civilian vehicles follow UK traffic rules.",
                    "presidential": "Presidential motorcades have special privileges but yield to emergency vehicles."
                }
            })
        
        return default_config
    
    def calculate_scores(self, vehicles: List[Vehicle]) -> Dict[str, float]:
        """Calculate priority scores for all vehicles."""
        scores = {}
        
        for vehicle in vehicles:
            # Base priority
            base_score = self.config["base_priority"].get(vehicle.kind.value, 10)
            
            # Lights/siren bonus
            lights_bonus = self.config["bonuses"]["lights"] if vehicle.has_right_of_way_signal else 0
            
            # Proximity bonus (closer = higher score)
            proximity_bonus = (20 - vehicle.distance_to_conflict_m) * self.config["bonuses"]["proximity_weight"]
            proximity_bonus = max(0, proximity_bonus)  # Don't go negative
            
            # Arrival time bonus (sooner = higher score)
            arrival_bonus = (10 - vehicle.arrival_time_s) * self.config["bonuses"]["arrival_weight"]
            arrival_bonus = max(0, arrival_bonus)  # Don't go negative
            
            # Total score
            total_score = base_score + lights_bonus + proximity_bonus + arrival_bonus
            scores[vehicle.id] = round(total_score, 1)
        
        return scores
    
    def apply_tie_breaks(self, vehicles: List[Vehicle], scores: Dict[str, float]) -> List[str]:
        """Apply tie-breaking rules to determine final order."""
        # Sort by score first
        sorted_vehicles = sorted(vehicles, key=lambda v: scores[v.id], reverse=True)
        
        # Group vehicles with same scores
        score_groups = {}
        for vehicle in sorted_vehicles:
            score = scores[vehicle.id]
            if score not in score_groups:
                score_groups[score] = []
            score_groups[score].append(vehicle)
        
        # Apply tie breaks within each score group
        final_order = []
        for score in sorted(score_groups.keys(), reverse=True):
            group = score_groups[score]
            if len(group) == 1:
                final_order.append(group[0].id)
            else:
                # Apply tie breaks
                tied_order = self._resolve_ties(group)
                final_order.extend([v.id for v in tied_order])
        
        return final_order
    
    def _resolve_ties(self, vehicles: List[Vehicle]) -> List[Vehicle]:
        """Resolve ties using configured tie-breaking rules."""
        for tie_break in self.config["tie_breaks"]:
            if tie_break == "right_hand_rule":
                vehicles = self._apply_right_hand_rule(vehicles)
            elif tie_break == "left_hand_rule":
                vehicles = self._apply_left_hand_rule(vehicles)
            elif tie_break == "apparatus_mass":
                vehicles = self._apply_apparatus_mass(vehicles)
            elif tie_break == "arrival_time":
                vehicles = self._apply_arrival_time(vehicles)
        
        return vehicles
    
    def _apply_right_hand_rule(self, vehicles: List[Vehicle]) -> List[Vehicle]:
        """Apply right-hand traffic rule."""
        return sorted(vehicles, key=lambda v: v.bearing_deg)
    
    def _apply_left_hand_rule(self, vehicles: List[Vehicle]) -> List[Vehicle]:
        """Apply left-hand traffic rule."""
        return sorted(vehicles, key=lambda v: v.bearing_deg, reverse=True)
    
    def _apply_apparatus_mass(self, vehicles: List[Vehicle]) -> List[Vehicle]:
        """Apply apparatus mass rule (fire > ambulance > police > presidential > civilian)."""
        mass_order = {
            VehicleType.FIRE_ENGINE: 5,
            VehicleType.AMBULANCE: 4,
            VehicleType.POLICE: 3,
            VehicleType.PRESIDENTIAL: 2,
            VehicleType.CIVILIAN: 1
        }
        return sorted(vehicles, key=lambda v: mass_order[v.kind], reverse=True)
    
    def _apply_arrival_time(self, vehicles: List[Vehicle]) -> List[Vehicle]:
        """Apply arrival time rule (earlier = higher priority)."""
        return sorted(vehicles, key=lambda v: v.arrival_time_s)

class VisionLayer:
    """Computer vision layer (stub implementation)."""
    
    def analyze_image(self, image_path: str) -> List[Vehicle]:
        """Analyze image and return detected vehicles."""
        # This is a stub - replace with real CV implementation
        filename = os.path.basename(image_path).lower()
        
        if "emergency" in filename or "chaos" in filename:
            return [
                Vehicle("ambulance_1", VehicleType.AMBULANCE, True, 8.0, 2.0, 45.0),
                Vehicle("fire_engine_1", VehicleType.FIRE_ENGINE, True, 12.0, 3.5, 135.0),
                Vehicle("police_1", VehicleType.POLICE, True, 15.0, 4.0, 225.0)
            ]
        elif "presidential" in filename or "escort" in filename:
            return [
                Vehicle("presidential_1", VehicleType.PRESIDENTIAL, True, 10.0, 2.5, 90.0),
                Vehicle("police_1", VehicleType.POLICE, True, 8.0, 2.0, 45.0),
                Vehicle("police_2", VehicleType.POLICE, True, 12.0, 3.0, 135.0)
            ]
        elif "civilian" in filename:
            return [
                Vehicle("civilian_1", VehicleType.CIVILIAN, False, 5.0, 1.5, 0.0),
                Vehicle("civilian_2", VehicleType.CIVILIAN, False, 8.0, 2.0, 90.0)
            ]
        else:
            # Default mixed scenario
            return [
                Vehicle("ambulance_1", VehicleType.AMBULANCE, True, 8.0, 2.0, 45.0),
                Vehicle("civilian_1", VehicleType.CIVILIAN, False, 5.0, 1.5, 0.0),
                Vehicle("police_1", VehicleType.POLICE, False, 12.0, 3.0, 135.0)
            ]

class LLMAdapter:
    """LLM adapter for reasoning validation."""
    
    def __init__(self):
        self.base_url = os.getenv("LLM_BASE_URL")
        self.api_key = os.getenv("LLM_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gpt-4")
        self.enabled = bool(self.base_url and self.api_key)
    
    def cross_check_decision(self, vehicles: List[Vehicle], rule_output: PolicyOutput) -> Optional[PolicyOutput]:
        """Cross-check rule engine decision with LLM reasoning."""
        if not self.enabled:
            return None
        
        try:
            # Prepare context for LLM
            context = self._prepare_context(vehicles, rule_output)
            
            # Call LLM
            response = self._call_llm(context)
            
            # Parse and validate response
            llm_output = self._parse_llm_response(response)
            
            return llm_output if llm_output else rule_output
            
        except Exception as e:
            print(f"LLM cross-check failed: {e}")
            return rule_output
    
    def _prepare_context(self, vehicles: List[Vehicle], rule_output: PolicyOutput) -> str:
        """Prepare context for LLM analysis."""
        context = f"""
Analyze this emergency intersection scenario and validate the priority decision:

VEHICLES DETECTED:
"""
        for vehicle in vehicles:
            context += f"- {vehicle.id}: {vehicle.kind.value}, lights={'ON' if vehicle.has_right_of_way_signal else 'OFF'}, distance={vehicle.distance_to_conflict_m}m, arrival={vehicle.arrival_time_s}s, bearing={vehicle.bearing_deg}°\n"
        
        context += f"""
RULE ENGINE DECISION:
Priority Order: {rule_output.ordered_ids}
Scores: {rule_output.scores}
Reasoning: {rule_output.reasoning}

Please validate this decision and return a JSON response with the same structure:
{{
    "ordered_ids": ["vehicle_id1", "vehicle_id2", ...],
    "scores": {{"vehicle_id1": score1, "vehicle_id2": score2, ...}},
    "reasoning": "explanation of the decision",
    "legality_notes": "legal considerations",
    "confidence": 0.95,
    "requires_human_confirmation": false
}}
"""
        return context
    
    def _call_llm(self, context: str) -> str:
        """Call the LLM API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert in traffic law and emergency vehicle priority systems. Analyze intersection scenarios and provide validated priority decisions."},
                {"role": "user", "content": context}
            ],
            "temperature": 0.1
        }
        
        response = requests.post(f"{self.base_url}/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _parse_llm_response(self, response: str) -> Optional[PolicyOutput]:
        """Parse and validate LLM response."""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["ordered_ids", "scores", "reasoning", "legality_notes"]
            for field in required_fields:
                if field not in data:
                    return None
            
            return PolicyOutput(**data)
            
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

class EmergencyIntersectionAnalyzer:
    """Main analyzer that orchestrates the entire process."""
    
    def __init__(self, jurisdiction: str = "us", use_llm: bool = False):
        self.rule_engine = RuleEngine(jurisdiction)
        self.vision_layer = VisionLayer()
        self.llm_adapter = LLMAdapter() if use_llm else None
    
    def analyze_image(self, image_path: str) -> PolicyOutput:
        """Analyze an intersection image and return priority decision."""
        # Step 1: Vision analysis
        vehicles = self.vision_layer.analyze_image(image_path)
        
        # Step 2: Rule engine scoring
        scores = self.rule_engine.calculate_scores(vehicles)
        ordered_ids = self.rule_engine.apply_tie_breaks(vehicles, scores)
        
        # Step 3: Prepare rule engine output
        rule_output = PolicyOutput(
            ordered_ids=ordered_ids,
            scores=scores,
            reasoning="Scores blend emergency level, right-of-way signals, proximity and arrival.",
            legality_notes=self.rule_engine.config["legal_notes"]["emergency"],
            confidence=1.0,
            requires_human_confirmation=False
        )
        
        # Step 4: Optional LLM cross-check
        if self.llm_adapter and self.llm_adapter.enabled:
            llm_output = self.llm_adapter.cross_check_decision(vehicles, rule_output)
            if llm_output:
                return llm_output
        
        return rule_output

# FastAPI Application
app = FastAPI(
    title="Emergency Intersection Priority System",
    description="AI-powered emergency vehicle priority analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer = EmergencyIntersectionAnalyzer()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Emergency Intersection Priority System",
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=PolicyOutput)
async def analyze_intersection(
    file: UploadFile = File(...),
    jurisdiction: str = "us",
    use_llm: bool = False
):
    """Analyze an intersection image and return priority decision."""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Create analyzer with specified parameters
        local_analyzer = EmergencyIntersectionAnalyzer(jurisdiction, use_llm)
        
        # Analyze image
        result = local_analyzer.analyze_image(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def create_visualization(ordered_ids: List[str], scores: Dict[str, float], title: str = "Priority Analysis") -> str:
    """Create a visualization of the priority analysis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Priority scores bar chart
    vehicles = ordered_ids
    vehicle_scores = [scores[vid] for vid in vehicles]
    
    bars = ax1.bar(range(len(vehicles)), vehicle_scores, 
                   color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'])
    ax1.set_title(f"{title} - Priority Scores", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Vehicle Priority Order", fontsize=12)
    ax1.set_ylabel("Priority Score", fontsize=12)
    ax1.set_xticks(range(len(vehicles)))
    ax1.set_xticklabels([vid.replace('_', ' ').title() for vid in vehicles], rotation=45)
    
    # Add score values on bars
    for i, (bar, score) in enumerate(zip(bars, vehicle_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Priority order flow
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
    for i, (vehicle, color) in enumerate(zip(vehicles, colors)):
        ax2.add_patch(patches.Rectangle((i*2, 0), 1.5, 1, facecolor=color, edgecolor='black'))
        ax2.text(i*2 + 0.75, 0.5, f"#{i+1}\n{vehicle.replace('_', ' ').title()}", 
                ha='center', va='center', fontweight='bold', fontsize=10)
    
    ax2.set_xlim(-0.5, len(vehicles)*2 - 0.5)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_title("Priority Order Flow", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add arrows between priority boxes
    for i in range(len(vehicles)-1):
        ax2.annotate('', xy=(i*2 + 1.75, 0.5), xytext=(i*2 + 1.5, 0.5),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.tight_layout()
    
    # Convert to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

@app.get("/visualize")
async def get_visualization(
    ordered_ids: str,
    scores: str,
    title: str = "Priority Analysis"
):
    """Get visualization of priority analysis."""
    try:
        # Parse parameters
        vehicle_list = ordered_ids.split(',')
        scores_dict = json.loads(scores)
        
        # Create visualization
        image_base64 = create_visualization(vehicle_list, scores_dict, title)
        
        return {
            "visualization": image_base64,
            "format": "base64_png"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

def main():
    """CLI interface for the analyzer."""
    parser = argparse.ArgumentParser(description="Emergency Intersection Priority System")
    parser.add_argument("--image", required=True, help="Path to intersection image")
    parser.add_argument("--jurisdiction", default="us", choices=["us", "uk"], help="Traffic jurisdiction")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM cross-check")
    parser.add_argument("--server", action="store_true", help="Start as web server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    if args.server:
        # Start web server
        uvicorn.run(app, host="0.0.0.0", port=args.port, reload=True)
    else:
        # CLI analysis
        analyzer = EmergencyIntersectionAnalyzer(args.jurisdiction, args.use_llm)
        result = analyzer.analyze_image(args.image)
        
        print("🚨 Emergency Intersection Priority Analysis")
        print("=" * 50)
        print(f"Image: {args.image}")
        print(f"Jurisdiction: {args.jurisdiction.upper()}")
        print(f"LLM Enabled: {args.use_llm}")
        print()
        print("📊 Results:")
        print(f"Priority Order: {' → '.join(result.ordered_ids)}")
        print(f"Scores: {result.scores}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Human Confirmation Required: {result.requires_human_confirmation}")
        print()
        print("💭 Reasoning:")
        print(result.reasoning)
        print()
        print("⚖️ Legal Notes:")
        print(result.legality_notes)

if __name__ == "__main__":
    main()
