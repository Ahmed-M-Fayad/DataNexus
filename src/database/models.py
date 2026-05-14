import enum
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Enums ─────────────────────────────────────────────────────────────────────

class SourceType(str, enum.Enum):
    postgresql = "postgresql"
    mysql      = "mysql"
    csv        = "csv"
    json       = "json"
    parquet    = "parquet"


class RunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    pass_   = "pass"   # trailing underscore avoids conflict with the Python keyword
    fail    = "fail"
    error   = "error"


class Severity(str, enum.Enum):
    critical = "critical"
    high     = "high"
    medium   = "medium"
    low      = "low"


class CheckStatus(str, enum.Enum):
    pass_  = "pass"   # trailing underscore avoids conflict with the Python keyword
    fail   = "fail"
    error  = "error"
    skip   = "skip"


class AlertChannel(str, enum.Enum):
    email = "email"
    slack = "slack"


class AlertStatus(str, enum.Enum):
    pending        = "pending"
    sent           = "sent"
    failed         = "failed"
    partially_sent = "partially_sent"


# ── Table 1: data_sources ────────────────────────────────────────────────────

class DataSource(Base):
    """Where data comes from — a database connection or a file path."""
    __tablename__ = "data_sources"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    name              = Column(String(255), nullable=False, unique=True)
    source_type       = Column(Enum(SourceType), nullable=False)
    connection_string = Column(Text, nullable=True)
    description       = Column(Text, nullable=True)
    is_active         = Column(Boolean, default=True, nullable=False)
    created_at        = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at        = Column(DateTime, default=datetime.utcnow,
                               onupdate=datetime.utcnow, nullable=False)

    datasets = relationship("Dataset", back_populates="source", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<DataSource id={self.id} name={self.name!r} type={self.source_type}>"


# ── Table 2: datasets ─────────────────────────────────────────────────────────

class Dataset(Base):
    """One table or file under a DataSource."""
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("source_id", "schema_name", "table_name",
                         name="uq_dataset_source_schema_table"),
    )

    id          = Column(Integer, primary_key=True, autoincrement=True)
    source_id   = Column(Integer, ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False)
    schema_name = Column(String(255), nullable=True)
    table_name  = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    is_active   = Column(Boolean, default=True, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)

    source             = relationship("DataSource", back_populates="datasets")
    profiles           = relationship("DataProfile", back_populates="dataset",
                                      cascade="all, delete-orphan")
    validation_configs = relationship("ValidationConfig", back_populates="dataset",
                                      cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Dataset id={self.id} table={self.schema_name}.{self.table_name}>"


# ── Table 3: data_profiles ────────────────────────────────────────────────────

class DataProfile(Base):
    """Statistical snapshot of a dataset at a point in time."""
    __tablename__ = "data_profiles"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id   = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    row_count    = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    profile_json = Column(Text, nullable=False)
    profiled_at  = Column(DateTime, default=datetime.utcnow, nullable=False)

    dataset = relationship("Dataset", back_populates="profiles")

    def __repr__(self) -> str:
        return (f"<DataProfile id={self.id} dataset_id={self.dataset_id} "
                f"rows={self.row_count} at={self.profiled_at}>")


# ── Table 4: validation_configs ───────────────────────────────────────────────

class ValidationConfig(Base):
    """Named set of validation rules for a dataset, stored as YAML text."""
    __tablename__ = "validation_configs"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id        = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    name              = Column(String(255), nullable=False)
    config_yaml       = Column(Text, nullable=False)
    schedule_cron     = Column(String(100), nullable=True)
    quality_threshold = Column(Float, default=0.95, nullable=False)  # 0.0–1.0
    alert_on_failure  = Column(Boolean, default=True, nullable=False)
    alert_channels    = Column(String(255), nullable=True)  # comma-separated, e.g. "email,slack"
    is_active         = Column(Boolean, default=True, nullable=False)
    created_at        = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at        = Column(DateTime, default=datetime.utcnow,
                               onupdate=datetime.utcnow, nullable=False)

    dataset         = relationship("Dataset", back_populates="validation_configs")
    validation_runs = relationship("ValidationRun", back_populates="config",
                                   cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ValidationConfig id={self.id} name={self.name!r}>"


# ── Table 5: test_definitions ─────────────────────────────────────────────────

class TestDefinition(Base):
    """Reusable named check templates (seeded with built-in check types)."""
    __tablename__ = "test_definitions"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    name                = Column(String(255), nullable=False, unique=True)
    category            = Column(String(100), nullable=False)  # e.g. "completeness", "validity"
    description         = Column(Text, nullable=True)
    implementation_code = Column(Text, nullable=True)
    is_builtin          = Column(Boolean, default=True, nullable=False)
    created_at          = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<TestDefinition id={self.id} name={self.name!r}>"


# ── Table 6: validation_runs ─────────────────────────────────────────────────

class ValidationRun(Base):
    """One execution of a ValidationConfig. Status: pending → running → pass/fail/error."""
    __tablename__ = "validation_runs"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    config_id     = Column(Integer, ForeignKey("validation_configs.id", ondelete="SET NULL"),
                           nullable=True)
    status        = Column(
        Enum(RunStatus, values_callable=lambda obj: [e.value for e in obj]),
        default=RunStatus.pending,
        nullable=False,
    )
    quality_score = Column(Float, nullable=True)   # 0.0–100.0, set after run
    triggered_by  = Column(String(100), nullable=True)  # "api" | "cli" | "airflow"
    started_at    = Column(DateTime, nullable=True)
    finished_at   = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)

    config             = relationship("ValidationConfig", back_populates="validation_runs")
    validation_results = relationship("ValidationResult", back_populates="run",
                                      cascade="all, delete-orphan")
    alerts             = relationship("Alert", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ValidationRun id={self.id} status={self.status} score={self.quality_score}>"


# ── Table 7: validation_results ──────────────────────────────────────────────

class ValidationResult(Base):
    """Per-check outcome inside one ValidationRun."""
    __tablename__ = "validation_results"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    run_id         = Column(Integer, ForeignKey("validation_runs.id", ondelete="CASCADE"),
                            nullable=False)
    check_name     = Column(String(255), nullable=False)
    column_name    = Column(String(255), nullable=True)
    check_type     = Column(String(100), nullable=False)
    status         = Column(
        Enum(CheckStatus, values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
    )
    severity       = Column(Enum(Severity), nullable=False)
    expected_value = Column(Text, nullable=True)
    actual_value   = Column(Text, nullable=True)
    failing_rows   = Column(Integer, nullable=True)
    total_rows     = Column(Integer, nullable=True)
    error_message  = Column(Text, nullable=True)
    executed_at    = Column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("ValidationRun", back_populates="validation_results")

    def __repr__(self) -> str:
        return f"<ValidationResult id={self.id} check={self.check_name!r} status={self.status}>"


# ── Table 8: alerts ───────────────────────────────────────────────────────────

class Alert(Base):
    """Dispatched notification linked to a failed ValidationRun."""
    __tablename__ = "alerts"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_id          = Column(Integer, ForeignKey("validation_runs.id", ondelete="CASCADE"),
                             nullable=False)
    channel         = Column(Enum(AlertChannel), nullable=False)
    alert_type      = Column(String(100), nullable=False)  # e.g. "validation_failure"
    severity        = Column(Enum(Severity), nullable=False)
    message         = Column(Text, nullable=False)
    status          = Column(Enum(AlertStatus), default=AlertStatus.pending, nullable=False)
    acknowledged    = Column(Boolean, default=False, nullable=False)
    acknowledged_by = Column(String(255), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    sent_at         = Column(DateTime, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("ValidationRun", back_populates="alerts")

    def __repr__(self) -> str:
        return (f"<Alert id={self.id} channel={self.channel} "
                f"status={self.status} ack={self.acknowledged}>")
