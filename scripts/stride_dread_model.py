"""
STRIDE-DREAD Threat Modeling for Kinect v1 + Raspberry Pi 4 IoT Edge Node
Paper 1 Component — Section III of the paper

STRIDE categories:
  S = Spoofing Identity
  T = Tampering with Data
  R = Repudiation
  I = Information Disclosure
  D = Denial of Service
  E = Elevation of Privilege

DREAD scoring (each 0-10):
  D = Damage potential
  R = Reproducibility
  E = Exploitability
  A = Affected users
  D = Discoverability
  Score = (D+R+E+A+D) / 5
"""

import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
import datetime

@dataclass
class Threat:
    threat_id: str
    layer: str              # Device / Network / Application
    stride_category: str    # S/T/R/I/D/E
    component: str          # Kinect RGB / Kinect Depth / Kinect Audio / Pi CPU / Network
    stream: str             # RGB / Depth / Audio / Network / System
    description: str
    attack_vector: str
    # DREAD scores
    damage: int
    reproducibility: int
    exploitability: int
    affected_users: int
    discoverability: int
    mitre_attack: str       # MITRE ATT&CK for ICS ID if applicable
    mitigation: str

    @property
    def dread_score(self) -> float:
        return round(
            (self.damage + self.reproducibility + self.exploitability +
             self.affected_users + self.discoverability) / 5.0, 2
        )

    @property
    def risk_level(self) -> str:
        score = self.dread_score
        if score >= 8.0:
            return "CRITICAL"
        elif score >= 6.0:
            return "HIGH"
        elif score >= 4.0:
            return "MEDIUM"
        else:
            return "LOW"


def build_kinect_pi4_threat_model() -> List[Threat]:
    """
    18 distinct threat vectors for Kinect v1 + Pi 4 trimodal edge node.
    First published STRIDE-DREAD model for this specific architecture.
    """
    threats = [
        # ============================================================
        # DEVICE LAYER THREATS
        # ============================================================

        # D1 — Physical Tampering
        Threat(
            threat_id="DL-T01",
            layer="Device",
            stride_category="T",
            component="Raspberry Pi 4 Hardware",
            stream="System",
            description="Physical access to USB ports enables insertion of malicious "
                        "USB device or direct memory access via DMA attack",
            attack_vector="Physical proximity, USB DMA (USB-C port on Pi 4)",
            damage=9, reproducibility=7, exploitability=6,
            affected_users=10, discoverability=5,
            mitre_attack="T0857",
            mitigation="SENTINEL-IoT physical enclosure with tamper-evident seals; "
                       "USB port locking via udev rules blacklisting unknown VIDs"
        ),

        # D2 — Kinect USB Spoofing
        Threat(
            threat_id="DL-T02",
            layer="Device",
            stride_category="S",
            component="Kinect USB Interface",
            stream="RGB+Depth+Audio",
            description="Adversary disconnects Kinect and attaches spoofed USB device "
                        "presenting same VID:PID (045e:02ae) to inject fake sensor data",
            attack_vector="USB VID:PID cloning via hardware emulator (e.g., Facedancer)",
            damage=8, reproducibility=5, exploitability=4,
            affected_users=9, discoverability=4,
            mitre_attack="T0862",
            mitigation="ECDSA-signed sensor handshake; hardware serial binding in "
                       "authentication module; USB device fingerprinting via timing analysis"
        ),

        # D3 — Firmware Tampering
        Threat(
            threat_id="DL-T03",
            layer="Device",
            stride_category="T",
            component="Raspberry Pi OS / MicroSD",
            stream="System",
            description="Attacker removes MicroSD card and modifies OS or application "
                        "code to insert backdoor or disable IDS components",
            attack_vector="Physical card removal; dd imaging + hex editing + reinstall",
            damage=10, reproducibility=8, exploitability=7,
            affected_users=10, discoverability=3,
            mitre_attack="T0839",
            mitigation="dm-verity filesystem integrity checking; SENTINEL-IoT "
                       "boot-time hash verification of all IDS binaries"
        ),

        # D4 — Depth Stream Biometric Disclosure
        Threat(
            threat_id="DL-T04",
            layer="Device",
            stride_category="I",
            component="Kinect Depth Sensor",
            stream="Depth",
            description="Raw 320x240 depth frames contain body silhouette and gait "
                        "patterns sufficient to identify individuals (biometric data "
                        "under GDPR Article 4(14)) without any RGB data",
            attack_vector="Network capture of unencrypted depth stream; "
                         "local storage exfiltration",
            damage=8, reproducibility=9, exploitability=8,
            affected_users=10, discoverability=6,
            mitre_attack="T0882",
            mitigation="Depth-Silhouette Privacy Module: morphological erosion + "
                       "calibrated Gaussian noise before any transmission or storage"
        ),

        # D5 — Audio Eavesdropping
        Threat(
            threat_id="DL-T05",
            layer="Device",
            stride_category="I",
            component="Kinect 4-Microphone Array",
            stream="Audio",
            description="16kHz audio capture from four-microphone beamforming array "
                        "can record private conversations in monitored space",
            attack_vector="Unencrypted audio buffer in memory; storage capture; "
                         "network stream interception",
            damage=7, reproducibility=8, exploitability=7,
            affected_users=9, discoverability=5,
            mitre_attack="T0882",
            mitigation="Audio feature extraction only (MFCC) — raw audio never "
                       "stored or transmitted; on-device anonymisation"
        ),

        # D6 — Side Channel Power Analysis
        Threat(
            threat_id="DL-T06",
            layer="Device",
            stride_category="I",
            component="Raspberry Pi 4 Power Supply",
            stream="System",
            description="Power consumption analysis of Pi 4 during ECDSA operations "
                        "may leak private key bits via simple power analysis (SPA)",
            attack_vector="Current probe on 5V power rail; oscilloscope capture "
                         "during authentication handshakes",
            damage=9, reproducibility=4, exploitability=3,
            affected_users=10, discoverability=2,
            mitre_attack="T0845",
            mitigation="Constant-time ECDSA implementation (libgcrypt constant-time); "
                       "noise injection via switched-mode power supply decoupling"
        ),

        # ============================================================
        # NETWORK LAYER THREATS
        # ============================================================

        # N1 — DDoS UDP Flood
        Threat(
            threat_id="NL-T01",
            layer="Network",
            stride_category="D",
            component="Pi 4 Network Interface",
            stream="Network",
            description="High-volume UDP flood targeting Pi 4 exhausts network buffer "
                        "and CPU preventing IDS from processing sensor streams",
            attack_vector="hping3 UDP flood; Mirai botnet coordinated attack",
            damage=7, reproducibility=9, exploitability=9,
            affected_users=10, discoverability=8,
            mitre_attack="T0814",
            mitigation="RCAF Engine rate-limiting; iptables flood mitigation rules; "
                       "SENTINEL-IoT DDoS detection via packet rate anomaly"
        ),

        # N2 — MQTT MITM
        Threat(
            threat_id="NL-T02",
            layer="Network",
            stride_category="T",
            component="Mosquitto MQTT Broker",
            stream="Network",
            description="Unencrypted MQTT on port 1883 allows attacker to intercept "
                        "and modify sensor data or IDS alerts in transit",
            attack_vector="ARP poisoning (Ettercap); TCP stream injection",
            damage=8, reproducibility=8, exploitability=8,
            affected_users=9, discoverability=8,
            mitre_attack="T0830",
            mitigation="MQTT over TLS 1.3; ECDSA-signed message payloads; "
                       "certificate pinning at subscriber"
        ),

        # N3 — ARP Spoofing
        Threat(
            threat_id="NL-T03",
            layer="Network",
            stride_category="S",
            component="Local Network (Wi-Fi/Ethernet)",
            stream="Network",
            description="ARP cache poisoning redirects Pi 4 traffic through attacker "
                        "machine enabling full MITM on all IoT communications",
            attack_vector="arp-spoof; Ettercap gratuitous ARP; arpwatch evasion",
            damage=8, reproducibility=9, exploitability=8,
            affected_users=9, discoverability=6,
            mitre_attack="T0830",
            mitigation="Static ARP entries for known nodes; SENTINEL-IoT ARP "
                       "anomaly detection in network monitor module"
        ),

        # N4 — Replay Attack on Authentication
        Threat(
            threat_id="NL-T04",
            layer="Network",
            stride_category="S",
            component="ECDSA Authentication Channel",
            stream="Network",
            description="Captured ECDSA authentication packets replayed to "
                        "impersonate legitimate node without possessing private key",
            attack_vector="Wireshark packet capture + Python replay script",
            damage=8, reproducibility=8, exploitability=7,
            affected_users=9, discoverability=5,
            mitre_attack="T0830",
            mitigation="Nonce-based challenge-response; timestamp binding with "
                       "5-second validity window; sequence number verification"
        ),

        # N5 — Network Reconnaissance
        Threat(
            threat_id="NL-T05",
            layer="Network",
            stride_category="I",
            component="Pi 4 Network Stack",
            stream="Network",
            description="Network scanning reveals Pi 4 device fingerprint, open ports, "
                        "and service versions enabling targeted attack preparation",
            attack_vector="nmap -sV -O; Shodan IoT scanning; banner grabbing",
            damage=5, reproducibility=10, exploitability=9,
            affected_users=8, discoverability=10,
            mitre_attack="T0840",
            mitigation="Port hardening; service version masking; SENTINEL-IoT "
                       "scan detection via connection rate monitoring"
        ),

        # N6 — MQTT Topic Injection
        Threat(
            threat_id="NL-T06",
            layer="Network",
            stride_category="T",
            component="Mosquitto MQTT Broker",
            stream="Network",
            description="Unauthenticated MQTT client publishes false sensor readings "
                        "to IDS alert topic suppressing or forging anomaly alerts",
            attack_vector="mosquitto_pub injection of fabricated JSON payloads",
            damage=8, reproducibility=9, exploitability=9,
            affected_users=10, discoverability=7,
            mitre_attack="T0856",
            mitigation="MQTT authentication with TLS client certificates; "
                       "topic ACL configuration in Mosquitto"
        ),

        # ============================================================
        # APPLICATION LAYER THREATS
        # ============================================================

        # A1 — OTA Firmware Hijack
        Threat(
            threat_id="AL-T01",
            layer="Application",
            stride_category="T",
            component="IDS Update Mechanism",
            stream="System",
            description="Malicious firmware or model update delivered via compromised "
                        "update channel replaces legitimate RCAF engine with backdoored model",
            attack_vector="MITM on HTTP update channel; DNS poisoning of update server",
            damage=10, reproducibility=6, exploitability=5,
            affected_users=10, discoverability=3,
            mitre_attack="T0839",
            mitigation="ECDSA-signed update packages; hash verification before install; "
                       "HTTPS with certificate pinning for update channel"
        ),

        # A2 — Gradient Inference Attack on Federated Model
        Threat(
            threat_id="AL-T02",
            layer="Application",
            stride_category="I",
            component="DepthGuard Federated Learning",
            stream="System",
            description="Malicious FL aggregator reconstructs private sensor data "
                        "from gradient updates shared by edge nodes",
            attack_vector="Gradient inversion attack (Geiping et al. 2020); "
                         "membership inference via shadow models",
            damage=8, reproducibility=5, exploitability=4,
            affected_users=10, discoverability=3,
            mitre_attack="T0882",
            mitigation="(ε=1.0, δ=1e-5) Differential Privacy via Gaussian noise "
                       "on gradients; gradient clipping (C=1.0)"
        ),

        # A3 — IDS Model Evasion
        Threat(
            threat_id="AL-T03",
            layer="Application",
            stride_category="E",
            component="RCAF Anomaly Detection Engine",
            stream="RGB+Depth",
            description="Adversary crafts physical intrusion behavior that stays below "
                        "detection threshold by moving slowly and using depth blind spots "
                        "at <0.8m and >4.0m (Kinect v1 range limits)",
            attack_vector="Kinect range exploitation; slow-movement adversarial walking",
            damage=8, reproducibility=6, exploitability=5,
            affected_users=10, discoverability=4,
            mitre_attack="T0858",
            mitigation="Cross-modal fusion — anomaly invisible in one stream "
                       "detected in another; context-aware threshold adaptation"
        ),

        # A4 — Log Repudiation
        Threat(
            threat_id="AL-T04",
            layer="Application",
            stride_category="R",
            component="IDS Alert Logging",
            stream="System",
            description="Attacker gains write access to log files and deletes or "
                        "modifies intrusion records to repudiate attack evidence",
            attack_vector="Local filesystem access via SSH credential compromise; "
                         "log rotation exploitation",
            damage=6, reproducibility=7, exploitability=6,
            affected_users=8, discoverability=5,
            mitre_attack="T0872",
            mitigation="Append-only log filesystem; SENTINEL-IoT log integrity "
                       "via HMAC chaining of log entries"
        ),

        # A5 — Python Process Privilege Escalation
        Threat(
            threat_id="AL-T05",
            layer="Application",
            stride_category="E",
            component="Python IDS Process",
            stream="System",
            description="Vulnerability in IDS Python dependencies (TF Lite, OpenCV, "
                        "libfreenect) exploited to escalate from sentinel user to root",
            attack_vector="CVE exploitation in outdated packages; "
                         "unsafe deserialization of model files",
            damage=9, reproducibility=4, exploitability=4,
            affected_users=10, discoverability=3,
            mitre_attack="T0874",
            mitigation="Principle of least privilege; IDS runs as sentinel user "
                       "not root; regular apt security updates; "
                       "model file signature verification"
        ),

        # A6 — RGB Stream Injection
        Threat(
            threat_id="AL-T06",
            layer="Application",
            stride_category="T",
            component="Kinect RGB Camera",
            stream="RGB",
            description="Attacker uses printed photo or video replay attack on Kinect "
                        "RGB camera to confuse MobileNetV3 motion detection",
            attack_vector="High-quality printed image held in front of camera; "
                         "video display spoofing RGB stream",
            damage=7, reproducibility=7, exploitability=6,
            affected_users=9, discoverability=6,
            mitre_attack="T0862",
            mitigation="Depth stream cross-validation — RGB-only anomaly must be "
                       "confirmed by corresponding depth occupancy change; "
                       "texture liveness detection"
        ),

        # A7 — DoS via Sensor Overflow
        Threat(
            threat_id="AL-T07",
            layer="Application",
            stride_category="D",
            component="Kinect USB + Pi 4 CPU",
            stream="RGB+Depth+Audio",
            description="Triggering continuous maximum-load sensor processing "
                        "(RGB+Depth+Audio at 30fps) exhausts Pi 4 CPU leaving no "
                        "capacity for IDS inference",
            attack_vector="Physical motion in front of Kinect causing continuous "
                         "high-complexity scene; combined with network flood",
            damage=6, reproducibility=7, exploitability=7,
            affected_users=10, discoverability=5,
            mitre_attack="T0814",
            mitigation="RCAF Engine frame sampling at 10fps under load; "
                       "CPU quota limits via systemd cgroups; "
                       "graceful degradation to depth-only mode"
        ),
    ]
    return threats


def generate_threat_report(threats: List[Threat], output_dir: str):
    """Generate threat model report in JSON and CSV formats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sort by DREAD score descending
    threats_sorted = sorted(threats, key=lambda t: t.dread_score, reverse=True)

    # Statistics
    stats = {
        "total_threats": len(threats),
        "critical": sum(1 for t in threats if t.risk_level == "CRITICAL"),
        "high": sum(1 for t in threats if t.risk_level == "HIGH"),
        "medium": sum(1 for t in threats if t.risk_level == "MEDIUM"),
        "low": sum(1 for t in threats if t.risk_level == "LOW"),
        "by_layer": {
            "Device": sum(1 for t in threats if t.layer == "Device"),
            "Network": sum(1 for t in threats if t.layer == "Network"),
            "Application": sum(1 for t in threats if t.layer == "Application")
        },
        "by_stride": {},
        "by_stream": {},
        "average_dread_score": round(
            sum(t.dread_score for t in threats) / len(threats), 2
        )
    }

    # STRIDE distribution
    for cat in ["S", "T", "R", "I", "D", "E"]:
        cat_names = {
            "S": "Spoofing", "T": "Tampering", "R": "Repudiation",
            "I": "Information Disclosure", "D": "Denial of Service",
            "E": "Elevation of Privilege"
        }
        stats["by_stride"][cat_names[cat]] = \
            sum(1 for t in threats if t.stride_category == cat)

    # Full report
    report = {
        "title": "STRIDE-DREAD Threat Model: Microsoft Kinect Xbox 360 (v1) + "
                 "Raspberry Pi 4 Trimodal IoT Edge Node",
        "generated": datetime.datetime.now().isoformat(),
        "architecture": {
            "edge_node": "Raspberry Pi 4B (4GB/8GB ARM Cortex-A72 @ 1.8GHz)",
            "sensor": "Microsoft Kinect Xbox 360 Model 1473 (Kinect v1)",
            "streams": ["RGB 640x480 @ 30fps", "Depth 320x240 @ 30fps",
                       "Audio 16kHz 4-microphone array"],
            "os": "Raspberry Pi OS 64-bit Bookworm",
            "connectivity": "Wi-Fi 802.11ac / Ethernet Gigabit"
        },
        "statistics": stats,
        "threats": [
            {
                **asdict(t),
                "dread_score": t.dread_score,
                "risk_level": t.risk_level
            }
            for t in threats_sorted
        ]
    }

    # Save JSON
    json_path = output_path / "stride_dread_threat_model.json"
    with open(str(json_path), "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved: {json_path}")

    # Save CSV
    csv_path = output_path / "stride_dread_threat_model.csv"
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Threat ID", "Layer", "STRIDE", "Component", "Stream",
            "Description", "Damage", "Reproducibility", "Exploitability",
            "Affected Users", "Discoverability", "DREAD Score",
            "Risk Level", "MITRE ATT&CK", "Mitigation"
        ])
        for t in threats_sorted:
            writer.writerow([
                t.threat_id, t.layer, t.stride_category, t.component,
                t.stream, t.description[:80] + "...",
                t.damage, t.reproducibility, t.exploitability,
                t.affected_users, t.discoverability,
                t.dread_score, t.risk_level, t.mitre_attack,
                t.mitigation[:60] + "..."
            ])
    print(f"CSV report saved: {csv_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("STRIDE-DREAD THREAT MODEL SUMMARY")
    print(f"Architecture: Kinect v1 (Model 1473) + Raspberry Pi 4")
    print("=" * 70)
    print(f"{'ID':<10} {'Layer':<12} {'STRIDE':<6} {'Stream':<12} "
          f"{'DREAD':>6} {'Risk':<10}")
    print("-" * 70)
    for t in threats_sorted:
        print(f"{t.threat_id:<10} {t.layer:<12} {t.stride_category:<6} "
              f"{t.stream:<12} {t.dread_score:>6.1f} {t.risk_level:<10}")
    print("-" * 70)
    print(f"Total: {stats['total_threats']} threats | "
          f"Critical: {stats['critical']} | High: {stats['high']} | "
          f"Medium: {stats['medium']} | Low: {stats['low']}")
    print(f"Average DREAD Score: {stats['average_dread_score']}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    threats = build_kinect_pi4_threat_model()
    report = generate_threat_report(
        threats,
        "/home/sentinel/sentinel_iot/paper1/results"
    )
    print(f"\nThreat model complete: {len(threats)} threats enumerated")

