from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from backend.database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, nullable=True, default="Unknown")
    cpu_usage = Column(Float, nullable=False, default=0.0)
    memory_usage = Column(Float, nullable=False, default=0.0)
    disk_io = Column(Float, nullable=False, default=0.0)
    network_traffic = Column(Float, nullable=False, default=0.0)
    prediction = Column(String, nullable=False, default="Normal")
    cause = Column(String, nullable=True, default="Not available")
    remediation = Column(String, nullable=True, default="None")
    latitude = Column(Float, nullable=True, default=0.0)
    longitude = Column(Float, nullable=True, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "device_id": self.device_id,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_io": self.disk_io,
            "network_traffic": self.network_traffic,
            "prediction": self.prediction,
            "cause": self.cause,
            "remediation": self.remediation,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

class QuarantinedDevice(Base):
    __tablename__ = "quarantined_devices"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, unique=True, index=True)
    reason = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "device_id": self.device_id,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }