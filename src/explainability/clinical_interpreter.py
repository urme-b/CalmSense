import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..logging_config import LoggerMixin


class StressLevel(Enum):
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class NormalRange:
    low: float
    high: float
    unit: str
    description: str
    stress_direction: str = "higher"  # "higher" or "lower" indicates
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None


@dataclass
class ClinicalFinding:
    feature: str
    value: float
    normal_range: NormalRange
    deviation: str  # "normal", "low", "high", "critical_low",
    stress_implication: str
    confidence: float
    interpretation: str


# Standard normal ranges based
NORMAL_RANGES = {
    # Heart Rate Variability -
    "hr_mean": NormalRange(
        low=60.0,
        high=100.0,
        unit="bpm",
        description="Mean heart rate",
        stress_direction="higher",
    ),
    "hrv_sdnn": NormalRange(
        low=50.0,
        high=100.0,
        unit="ms",
        description="Standard deviation of NN intervals",
        stress_direction="lower",
        critical_low=30.0,
    ),
    "hrv_rmssd": NormalRange(
        low=25.0,
        high=45.0,
        unit="ms",
        description="Root mean square of successive differences",
        stress_direction="lower",
        critical_low=15.0,
    ),
    "hrv_pnn50": NormalRange(
        low=5.0,
        high=50.0,
        unit="%",
        description="Percentage of NN50 intervals",
        stress_direction="lower",
        critical_low=2.0,
    ),
    "hrv_nn50": NormalRange(
        low=30.0,
        high=300.0,
        unit="count",
        description="Number of NN50 intervals",
        stress_direction="lower",
    ),
    # Heart Rate Variability -
    "hrv_lf_power": NormalRange(
        low=400.0,
        high=1800.0,
        unit="ms^2",
        description="Low frequency power (0.04-0.15 Hz)",
        stress_direction="higher",
    ),
    "hrv_hf_power": NormalRange(
        low=250.0,
        high=1500.0,
        unit="ms^2",
        description="High frequency power (0.15-0.4 Hz)",
        stress_direction="lower",
    ),
    "hrv_lf_hf_ratio": NormalRange(
        low=1.0,
        high=2.0,
        unit="ratio",
        description="LF/HF ratio (sympathovagal balance)",
        stress_direction="higher",
        critical_high=4.0,
    ),
    "hrv_vlf_power": NormalRange(
        low=200.0,
        high=2000.0,
        unit="ms^2",
        description="Very low frequency power (0.0033-0.04 Hz)",
        stress_direction="higher",
    ),
    # Electrodermal Activity
    "eda_mean": NormalRange(
        low=2.0,
        high=16.0,
        unit="μS",
        description="Mean skin conductance level",
        stress_direction="higher",
    ),
    "eda_std": NormalRange(
        low=0.5,
        high=3.0,
        unit="μS",
        description="Skin conductance variability",
        stress_direction="higher",
    ),
    "scr_count": NormalRange(
        low=0.0,
        high=20.0,
        unit="count/min",
        description="Skin conductance response frequency",
        stress_direction="higher",
    ),
    "scr_amplitude_mean": NormalRange(
        low=0.05,
        high=1.0,
        unit="μS",
        description="Mean SCR amplitude",
        stress_direction="higher",
    ),
    # Respiration
    "resp_rate": NormalRange(
        low=12.0,
        high=20.0,
        unit="breaths/min",
        description="Respiratory rate",
        stress_direction="higher",
        critical_high=30.0,
        critical_low=8.0,
    ),
    "resp_depth": NormalRange(
        low=0.3,
        high=0.8,
        unit="L",
        description="Respiratory depth (tidal volume estimate)",
        stress_direction="lower",
    ),
    "resp_variability": NormalRange(
        low=0.1,
        high=0.3,
        unit="CV",
        description="Respiratory rate variability",
        stress_direction="higher",
    ),
    # Temperature
    "temp_mean": NormalRange(
        low=32.0,
        high=35.0,
        unit="°C",
        description="Peripheral skin temperature",
        stress_direction="lower",
    ),
    "temp_variability": NormalRange(
        low=0.1,
        high=0.5,
        unit="°C",
        description="Temperature variability",
        stress_direction="higher",
    ),
    # Blood Volume Pulse
    "bvp_amplitude": NormalRange(
        low=0.5,
        high=2.0,
        unit="a.u.",
        description="Blood volume pulse amplitude",
        stress_direction="lower",
    ),
    # Electromyography
    "emg_mean": NormalRange(
        low=5.0,
        high=50.0,
        unit="μV",
        description="Mean muscle activity",
        stress_direction="higher",
    ),
    "emg_activation_count": NormalRange(
        low=0.0,
        high=10.0,
        unit="count/min",
        description="Muscle tension episodes",
        stress_direction="higher",
    ),
}


# Stress level thresholds based
STRESS_THRESHOLDS = {
    StressLevel.LOW: (-2.0, -1.0),  # 1-2 SD below mean
    StressLevel.NORMAL: (-1.0, 1.0),  # Within 1 SD
    StressLevel.ELEVATED: (1.0, 2.0),  # 1-2 SD above mean
    StressLevel.HIGH: (2.0, 3.0),  # 2-3 SD above mean
    StressLevel.VERY_HIGH: (3.0, float("inf")),  # >3 SD
}


class ClinicalInterpreter(LoggerMixin):
    def __init__(
        self,
        class_names: List[str] = None,
        normal_ranges: Optional[Dict[str, NormalRange]] = None,
        custom_thresholds: Optional[Dict] = None,
    ):

        self.class_names = class_names or ["Baseline", "Stress", "Amusement"]

        # Merge custom ranges with
        self.normal_ranges = NORMAL_RANGES.copy()
        if normal_ranges:
            self.normal_ranges.update(normal_ranges)

        self.stress_thresholds = custom_thresholds or STRESS_THRESHOLDS

        # Feature name mapping for
        self.feature_aliases = {
            "mean_hr": "hr_mean",
            "heart_rate": "hr_mean",
            "sdnn": "hrv_sdnn",
            "rmssd": "hrv_rmssd",
            "pnn50": "hrv_pnn50",
            "lf_power": "hrv_lf_power",
            "hf_power": "hrv_hf_power",
            "lf_hf": "hrv_lf_hf_ratio",
            "scl_mean": "eda_mean",
            "skin_conductance": "eda_mean",
            "respiratory_rate": "resp_rate",
            "breathing_rate": "resp_rate",
        }

    def _resolve_feature_name(self, name: str) -> str:

        # Try direct match first
        if name in self.normal_ranges:
            return name

        # Try lowercase
        lower_name = name.lower()
        if lower_name in self.normal_ranges:
            return lower_name

        # Try aliases
        if lower_name in self.feature_aliases:
            return self.feature_aliases[lower_name]

        # Try partial matches
        for alias, canonical in self.feature_aliases.items():
            if alias in lower_name or lower_name in alias:
                return canonical

        return name

    def assess_feature(
        self, feature_name: str, value: float, importance: Optional[float] = None
    ) -> ClinicalFinding:

        # Resolve feature name
        resolved_name = self._resolve_feature_name(feature_name)

        if resolved_name not in self.normal_ranges:
            return ClinicalFinding(
                feature=feature_name,
                value=value,
                normal_range=NormalRange(0, 0, "", "Unknown feature", ""),
                deviation="unknown",
                stress_implication="Unable to assess - feature not in database",
                confidence=0.0,
                interpretation=f"Feature '{feature_name}' not found in clinical database",
            )

        normal = self.normal_ranges[resolved_name]

        # Determine deviation
        if normal.critical_low and value < normal.critical_low:
            deviation = "critical_low"
        elif normal.critical_high and value > normal.critical_high:
            deviation = "critical_high"
        elif value < normal.low:
            deviation = "low"
        elif value > normal.high:
            deviation = "high"
        else:
            deviation = "normal"

        # Determine stress implication
        stress_implication = self._interpret_stress_implication(
            deviation, normal.stress_direction
        )

        # Calculate confidence based on
        if deviation == "normal":
            # How centered in normal
            range_center = (normal.high + normal.low) / 2
            range_width = normal.high - normal.low
            distance_from_center = abs(value - range_center) / (range_width / 2)
            confidence = 1.0 - min(distance_from_center, 1.0) * 0.5
        else:
            # How far from normal
            if value < normal.low:
                distance = (normal.low - value) / max(abs(normal.low), 1e-10)
            else:
                distance = (value - normal.high) / max(abs(normal.high), 1e-10)
            confidence = min(0.5 + distance, 1.0)

        # Boost confidence if feature
        if importance is not None:
            confidence = confidence * 0.7 + importance * 0.3

        # Generate interpretation
        interpretation = self._generate_interpretation(
            resolved_name, value, normal, deviation, stress_implication
        )

        return ClinicalFinding(
            feature=feature_name,
            value=value,
            normal_range=normal,
            deviation=deviation,
            stress_implication=stress_implication,
            confidence=confidence,
            interpretation=interpretation,
        )

    def _interpret_stress_implication(
        self, deviation: str, stress_direction: str
    ) -> str:

        if deviation == "normal":
            return "Within normal range - no stress indication"

        # Check if deviation is
        if deviation == "critical_low" or deviation == "critical_high":
            # For critical deviations, include
            if stress_direction == "higher":
                if deviation == "critical_high":
                    return "Critical stress elevation - requires immediate attention"
                else:
                    return "Critical low value - requires attention"
            else:  # lower indicates stress
                if deviation == "critical_low":
                    return "Critical stress indicator - requires immediate attention"
                else:
                    return "Critical high value - requires attention"

        # Check if deviation is
        if stress_direction == "higher":
            if deviation == "high":
                return "Elevated - consistent with stress response"
            else:
                return "Low - may indicate relaxation or parasympathetic dominance"
        else:  # lower indicates stress
            if deviation == "low":
                return "Reduced - consistent with stress response"
            else:
                return "Elevated - may indicate relaxation or recovery"

    def _generate_interpretation(
        self,
        feature_name: str,
        value: float,
        normal: NormalRange,
        deviation: str,
        stress_implication: str,
    ) -> str:

        lines = []

        # Feature description
        lines.append(f"{normal.description}: {value:.2f} {normal.unit}")

        # Normal range context
        lines.append(
            f"Normal range: {normal.low:.1f} - {normal.high:.1f} {normal.unit}"
        )

        # Deviation assessment
        if deviation == "normal":
            lines.append("Assessment: Within normal limits")
        elif deviation == "critical_low":
            lines.append(
                f"Assessment: CRITICALLY LOW (below {normal.critical_low} {normal.unit})"
            )
        elif deviation == "critical_high":
            lines.append(
                f"Assessment: CRITICALLY HIGH (above {normal.critical_high} {normal.unit})"
            )
        elif deviation == "low":
            pct_below = ((normal.low - value) / max(abs(normal.low), 1e-10)) * 100
            lines.append(
                f"Assessment: Below normal ({pct_below:.1f}% below lower limit)"
            )
        else:  # high
            pct_above = ((value - normal.high) / max(abs(normal.high), 1e-10)) * 100
            lines.append(
                f"Assessment: Above normal ({pct_above:.1f}% above upper limit)"
            )

        # Clinical significance
        if "hrv" in feature_name.lower() or "rmssd" in feature_name.lower():
            if deviation in ["low", "critical_low"]:
                lines.append(
                    "Clinical note: Reduced HRV is associated with increased "
                    "sympathetic activation and decreased parasympathetic tone, "
                    "commonly observed during acute stress."
                )
        elif "eda" in feature_name.lower() or "scr" in feature_name.lower():
            if deviation == "high":
                lines.append(
                    "Clinical note: Elevated electrodermal activity indicates "
                    "increased sympathetic arousal, consistent with stress response."
                )
        elif "lf_hf" in feature_name.lower():
            if deviation == "high":
                lines.append(
                    "Clinical note: Elevated LF/HF ratio suggests sympathetic "
                    "dominance over parasympathetic activity (Task Force, 1996)."
                )

        return "\n".join(lines)

    def interpret_prediction(
        self,
        prediction: int,
        probabilities: List[float],
        feature_values: Dict[str, float],
        feature_importances: Optional[Dict[str, float]] = None,
        shap_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        # Assess all features
        findings = []
        for feature, value in feature_values.items():
            importance = None
            if feature_importances:
                importance = feature_importances.get(feature)
            if shap_values:
                importance = abs(shap_values.get(feature, 0))

            finding = self.assess_feature(feature, value, importance)
            findings.append(finding)

        # Categorize findings
        stress_indicators = [
            f
            for f in findings
            if "stress" in f.stress_implication.lower() and f.deviation != "normal"
        ]
        normal_indicators = [f for f in findings if f.deviation == "normal"]
        critical_indicators = [f for f in findings if "critical" in f.deviation]

        # Determine overall stress level
        stress_score = self._compute_stress_score(findings)
        stress_level = self._categorize_stress_level(stress_score)

        # Generate summary
        summary = self._generate_summary(
            prediction,
            probabilities,
            stress_level,
            stress_indicators,
            normal_indicators,
            critical_indicators,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            stress_level, stress_indicators
        )

        report = {
            "prediction": {
                "class": prediction,
                "class_name": self.class_names[prediction],
                "probabilities": {
                    name: prob for name, prob in zip(self.class_names, probabilities)
                },
                "confidence": max(probabilities),
            },
            "stress_assessment": {
                "stress_score": stress_score,
                "stress_level": stress_level.value,
                "stress_indicators": len(stress_indicators),
                "normal_indicators": len(normal_indicators),
                "critical_indicators": len(critical_indicators),
            },
            "findings": [
                {
                    "feature": f.feature,
                    "value": f.value,
                    "unit": f.normal_range.unit,
                    "deviation": f.deviation,
                    "stress_implication": f.stress_implication,
                    "confidence": f.confidence,
                    "interpretation": f.interpretation,
                }
                for f in findings
            ],
            "summary": summary,
            "recommendations": recommendations,
            "disclaimer": (
                "This interpretation is for research purposes only and should not "
                "be used for clinical diagnosis. Always consult a healthcare "
                "professional for medical advice."
            ),
        }

        return report

    def _compute_stress_score(self, findings: List[ClinicalFinding]) -> float:

        if not findings:
            return 0.0

        scores = []
        for finding in findings:
            if finding.deviation == "unknown":
                continue

            # Base score on deviation
            if finding.deviation == "normal":
                score = 0.0
            elif (
                finding.deviation == "critical_high"
                or finding.deviation == "critical_low"
            ):
                score = 1.0
            elif finding.deviation in ["high", "low"]:
                # Check if deviation is
                normal = finding.normal_range
                if normal.stress_direction == "higher" and finding.deviation == "high":
                    score = 0.7
                elif normal.stress_direction == "lower" and finding.deviation == "low":
                    score = 0.7
                else:
                    score = 0.3  # Opposite direction

            scores.append(score * finding.confidence)

        return np.mean(scores) if scores else 0.0

    def _categorize_stress_level(self, score: float) -> StressLevel:

        if score < 0.2:
            return StressLevel.LOW
        elif score < 0.4:
            return StressLevel.NORMAL
        elif score < 0.6:
            return StressLevel.ELEVATED
        elif score < 0.8:
            return StressLevel.HIGH
        else:
            return StressLevel.VERY_HIGH

    def _generate_summary(
        self,
        prediction: int,
        probabilities: List[float],
        stress_level: StressLevel,
        stress_indicators: List[ClinicalFinding],
        normal_indicators: List[ClinicalFinding],
        critical_indicators: List[ClinicalFinding],
    ) -> str:

        lines = []

        # Prediction summary
        pred_name = self.class_names[prediction]
        pred_conf = max(probabilities) * 100
        lines.append(f"Predicted State: {pred_name} ({pred_conf:.1f}% confidence)")

        # Stress level
        stress_desc = {
            StressLevel.LOW: "minimal stress indicators",
            StressLevel.NORMAL: "physiological parameters within normal limits",
            StressLevel.ELEVATED: "some indicators of elevated stress",
            StressLevel.HIGH: "multiple indicators of significant stress",
            StressLevel.VERY_HIGH: "pronounced stress response across multiple markers",
        }
        lines.append(f"\nStress Level: {stress_level.value.upper()}")
        lines.append(f"Assessment: {stress_desc[stress_level]}")

        # Critical findings
        if critical_indicators:
            lines.append(f"\nCRITICAL FINDINGS ({len(critical_indicators)}):")
            for finding in critical_indicators[:3]:
                lines.append(
                    f"  - {finding.feature}: {finding.value:.2f} {finding.normal_range.unit}"
                )

        # Stress indicators
        if stress_indicators:
            lines.append(f"\nStress Indicators ({len(stress_indicators)}):")
            for finding in sorted(stress_indicators, key=lambda x: -x.confidence)[:5]:
                lines.append(
                    f"  - {finding.normal_range.description}: "
                    f"{finding.value:.2f} {finding.normal_range.unit} ({finding.deviation})"
                )

        # Normal findings
        if normal_indicators:
            lines.append(f"\nNormal Findings ({len(normal_indicators)}):")
            for finding in normal_indicators[:3]:
                lines.append(
                    f"  - {finding.normal_range.description}: "
                    f"{finding.value:.2f} {finding.normal_range.unit}"
                )

        return "\n".join(lines)

    def _generate_recommendations(
        self, stress_level: StressLevel, stress_indicators: List[ClinicalFinding]
    ) -> List[str]:

        recommendations = []

        if stress_level in [StressLevel.HIGH, StressLevel.VERY_HIGH]:
            recommendations.extend(
                [
                    "Consider immediate stress reduction techniques (deep breathing, progressive muscle relaxation)",
                    "Evaluate environmental stressors and consider removing or reducing exposure",
                    "Monitor physiological markers over time to track recovery",
                ]
            )

            # Specific recommendations based on
            hrv_low = any(
                "hrv" in f.feature.lower() and f.deviation in ["low", "critical_low"]
                for f in stress_indicators
            )
            if hrv_low:
                recommendations.append(
                    "Low HRV detected: Consider heart rate variability biofeedback training"
                )

            eda_high = any(
                "eda" in f.feature.lower() and f.deviation == "high"
                for f in stress_indicators
            )
            if eda_high:
                recommendations.append(
                    "Elevated electrodermal activity: Relaxation exercises may help reduce sympathetic arousal"
                )

        elif stress_level == StressLevel.ELEVATED:
            recommendations.extend(
                [
                    "Mild stress elevation detected - continue monitoring",
                    "Consider preventive stress management strategies",
                    "Maintain regular breaks and recovery periods",
                ]
            )

        elif stress_level in [StressLevel.LOW, StressLevel.NORMAL]:
            recommendations.extend(
                [
                    "Physiological markers indicate good adaptation",
                    "Continue current wellness practices",
                    "Regular monitoring recommended for longitudinal tracking",
                ]
            )

        return recommendations

    def compare_sessions(
        self,
        session1_features: Dict[str, float],
        session2_features: Dict[str, float],
        session1_name: str = "Session 1",
        session2_name: str = "Session 2",
    ) -> pd.DataFrame:

        all_features = set(session1_features.keys()) | set(session2_features.keys())

        comparison_data = []
        for feature in all_features:
            val1 = session1_features.get(feature)
            val2 = session2_features.get(feature)

            if val1 is None or val2 is None:
                continue

            resolved = self._resolve_feature_name(feature)
            normal = self.normal_ranges.get(resolved)

            change = val2 - val1
            pct_change = (change / val1 * 100) if val1 != 0 else 0

            # Determine if change is
            if normal:
                if normal.stress_direction == "higher":
                    stress_change = "increased" if change > 0 else "decreased"
                else:
                    stress_change = "increased" if change < 0 else "decreased"
            else:
                stress_change = "unknown"

            comparison_data.append(
                {
                    "feature": feature,
                    session1_name: val1,
                    session2_name: val2,
                    "change": change,
                    "percent_change": pct_change,
                    "stress_trend": stress_change,
                    "unit": normal.unit if normal else "",
                }
            )

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("percent_change", key=abs, ascending=False)

        return df

    def get_feature_info(self, feature_name: str) -> Optional[Dict]:

        resolved = self._resolve_feature_name(feature_name)

        if resolved not in self.normal_ranges:
            return None

        normal = self.normal_ranges[resolved]

        return {
            "feature": resolved,
            "description": normal.description,
            "normal_range": f"{normal.low} - {normal.high} {normal.unit}",
            "unit": normal.unit,
            "stress_direction": normal.stress_direction,
            "critical_low": normal.critical_low,
            "critical_high": normal.critical_high,
        }

    def list_supported_features(self) -> List[str]:

        return list(self.normal_ranges.keys())

    def add_custom_range(
        self,
        feature_name: str,
        low: float,
        high: float,
        unit: str,
        description: str,
        stress_direction: str = "higher",
        critical_low: Optional[float] = None,
        critical_high: Optional[float] = None,
    ) -> None:

        self.normal_ranges[feature_name] = NormalRange(
            low=low,
            high=high,
            unit=unit,
            description=description,
            stress_direction=stress_direction,
            critical_low=critical_low,
            critical_high=critical_high,
        )
        self.logger.info(f"Added custom range for: {feature_name}")
