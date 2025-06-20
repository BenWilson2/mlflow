import json
import time
from unittest import mock

import pytest

import mlflow.tracking.context.default_context
from mlflow.entities.span import LiveSpan
from mlflow.entities.trace_info_v2 import TraceInfoV2
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.constant import (
    TRACE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION_KEY,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceTagKey,
)
from mlflow.tracing.processor.mlflow_v2 import MlflowV2SpanProcessor
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_USER,
)
from mlflow.utils.os import is_windows
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.tracing.helper import (
    create_mock_otel_span,
    skip_when_testing_trace_sdk,
)

_TRACE_ID = 12345
_REQUEST_ID = f"tr-{_TRACE_ID}"


def test_on_start(monkeypatch):
    # Root span should create a new trace on start
    span = create_mock_otel_span(
        trace_id=_TRACE_ID, span_id=1, parent_id=None, start_time=5_000_000
    )
    trace_info = create_test_trace_info(_REQUEST_ID, 0)

    mock_client = mock.MagicMock()
    mock_client.start_trace.return_value = trace_info

    with mock.patch("mlflow.tracing.processor.mlflow_v2.TracingClient", return_value=mock_client):
        processor = MlflowV2SpanProcessor(span_exporter=mock.MagicMock())

    processor.on_start(span)

    expected_metadata = {
        TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION),
        MLFLOW_USER: mock.ANY,
        MLFLOW_SOURCE_NAME: mock.ANY,
        MLFLOW_SOURCE_TYPE: "LOCAL",
    }
    if not IS_TRACING_SDK_ONLY:
        expected_metadata.update(
            {
                MLFLOW_GIT_BRANCH: mock.ANY,
                MLFLOW_GIT_COMMIT: mock.ANY,
                MLFLOW_GIT_REPO_URL: mock.ANY,
            }
        )

    mock_client.start_trace.assert_called_once_with(
        experiment_id=_get_experiment_id(),
        timestamp_ms=5,
        request_metadata=expected_metadata,
        tags={TraceTagKey.TRACE_NAME: "test_span"},
    )
    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(_REQUEST_ID)
    assert _REQUEST_ID in InMemoryTraceManager.get_instance()._traces

    # Child span should not create a new trace
    child_span = create_mock_otel_span(
        trace_id=_TRACE_ID, span_id=2, parent_id=1, start_time=8_000_000
    )
    mock_client.start_trace.reset_mock()
    processor.on_start(child_span)

    mock_client.start_trace.assert_not_called()
    assert child_span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(_REQUEST_ID)


@pytest.mark.skipif(is_windows(), reason="Timestamp is not precise enough on Windows")
def test_on_start_adjust_span_timestamp_to_exclude_backend_latency(monkeypatch):
    monkeypatch.setenv("MLFLOW_TESTING", "false")
    trace_info = create_test_trace_info(_REQUEST_ID, 0)
    mock_client = mock.MagicMock()

    def _mock_start_trace(*args, **kwargs):
        time.sleep(0.5)  # Simulate backend latency
        return trace_info

    mock_client.start_trace.side_effect = _mock_start_trace

    with mock.patch("mlflow.tracing.processor.mlflow_v2.TracingClient", return_value=mock_client):
        processor = MlflowV2SpanProcessor(span_exporter=mock.MagicMock())

    original_start_time = time.time_ns()
    span = create_mock_otel_span(trace_id=_TRACE_ID, span_id=1, start_time=original_start_time)

    # make sure start_trace is invoked
    assert processor._trace_manager.get_mlflow_trace_id_from_otel_id(span.context.trace_id) is None
    processor.on_start(span)

    assert span.start_time > original_start_time
    # The span timestamp should not include the backend latency (0.5 second)
    assert time.time_ns() - span.start_time < 100_000_000  # 0.1 second


def test_on_start_with_experiment_id(monkeypatch):
    experiment_id = "test_experiment_id"
    span = create_mock_otel_span(
        trace_id=_TRACE_ID, span_id=1, parent_id=None, start_time=5_000_000
    )
    span.set_attribute(SpanAttributeKey.EXPERIMENT_ID, json.dumps(experiment_id))
    trace_info = create_test_trace_info(_REQUEST_ID, experiment_id=experiment_id)

    mock_client = mock.MagicMock()
    mock_client.start_trace.return_value = trace_info
    with mock.patch("mlflow.tracing.processor.mlflow_v2.TracingClient", return_value=mock_client):
        processor = MlflowV2SpanProcessor(span_exporter=mock.MagicMock())

    processor.on_start(span)

    mock_client.start_trace.assert_called_once_with(
        experiment_id=experiment_id,
        timestamp_ms=5,
        request_metadata=mock.ANY,
        tags={TraceTagKey.TRACE_NAME: "test_span"},
    )
    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(_REQUEST_ID)
    assert _REQUEST_ID in InMemoryTraceManager.get_instance()._traces


@skip_when_testing_trace_sdk
def test_on_start_during_model_evaluation():
    from mlflow.pyfunc.context import Context, set_prediction_context

    # Root span should create a new trace on start
    span = create_mock_otel_span(trace_id=_TRACE_ID, span_id=1)
    mock_client = mock.MagicMock()
    mock_client.start_trace.return_value = create_test_trace_info(_REQUEST_ID, 0)

    with mock.patch("mlflow.tracing.processor.mlflow_v2.TracingClient", return_value=mock_client):
        processor = MlflowV2SpanProcessor(span_exporter=mock.MagicMock())

    with set_prediction_context(Context(request_id=_REQUEST_ID, is_evaluate=True)):
        processor.on_start(span)

    mock_client.start_trace.assert_called_once()
    assert span.attributes.get(SpanAttributeKey.REQUEST_ID) == json.dumps(_REQUEST_ID)


@skip_when_testing_trace_sdk
def test_on_start_during_run(monkeypatch):
    span = create_mock_otel_span(
        trace_id=_TRACE_ID, span_id=1, parent_id=None, start_time=5_000_000
    )

    env_experiment_name = "env_experiment_id"
    run_experiment_name = "run_experiment_id"

    mlflow.create_experiment(env_experiment_name)
    run_experiment_id = mlflow.create_experiment(run_experiment_name)

    mlflow.set_experiment(experiment_name=env_experiment_name)
    trace_info = create_test_trace_info(_REQUEST_ID)
    mock_client = mock.MagicMock()
    mock_client.start_trace.return_value = trace_info

    with mock.patch("mlflow.tracing.processor.mlflow_v2.TracingClient", return_value=mock_client):
        processor = MlflowV2SpanProcessor(span_exporter=mock.MagicMock())

    with mlflow.start_run(experiment_id=run_experiment_id) as run:
        processor.on_start(span)
        expected_run_id = run.info.run_id

    expected_metadata = {
        TraceMetadataKey.SOURCE_RUN: expected_run_id,
        TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION),
        MLFLOW_USER: mock.ANY,
        MLFLOW_SOURCE_NAME: mock.ANY,
        MLFLOW_SOURCE_TYPE: "LOCAL",
    }
    if not IS_TRACING_SDK_ONLY:
        expected_metadata.update(
            {
                MLFLOW_GIT_BRANCH: mock.ANY,
                MLFLOW_GIT_COMMIT: mock.ANY,
                MLFLOW_GIT_REPO_URL: mock.ANY,
            }
        )

    mock_client.start_trace.assert_called_once_with(
        # expect experiment id to be from the run, not from the environment
        experiment_id=run_experiment_id,
        timestamp_ms=5,
        # expect run id to be set
        request_metadata=expected_metadata,
        tags=mock.ANY,
    )


def test_on_start_with_experiment_id_override(monkeypatch):
    mlflow.set_experiment(experiment_id=DEFAULT_EXPERIMENT_ID)

    mock_client = mock.MagicMock()
    mock_client.start_trace.return_value = create_test_trace_info(_REQUEST_ID, 0)

    mock_logger = mock.MagicMock()
    monkeypatch.setattr("mlflow.tracing.processor.mlflow_v2._logger", mock_logger)

    with mock.patch("mlflow.tracing.processor.mlflow_v2.TracingClient", return_value=mock_client):
        processor = MlflowV2SpanProcessor(
            span_exporter=mock.MagicMock(), experiment_id="another_experiment"
        )

    processor.on_start(create_mock_otel_span(trace_id=123, span_id=1))

    mock_client.start_trace.assert_called_once_with(
        experiment_id="another_experiment",
        timestamp_ms=mock.ANY,
        request_metadata=mock.ANY,
        tags=mock.ANY,
    )


def test_on_start_warns_default_experiment(monkeypatch):
    mlflow.set_experiment(experiment_id=DEFAULT_EXPERIMENT_ID)

    mock_client = mock.MagicMock()
    mock_client.start_trace.return_value = create_test_trace_info(_REQUEST_ID, 0)

    mock_logger = mock.MagicMock()
    monkeypatch.setattr("mlflow.tracing.processor.mlflow_v2._logger", mock_logger)

    with mock.patch("mlflow.tracing.processor.mlflow_v2.TracingClient", return_value=mock_client):
        processor = MlflowV2SpanProcessor(span_exporter=mock.MagicMock())

    processor.on_start(create_mock_otel_span(trace_id=123, span_id=1))
    processor.on_start(create_mock_otel_span(trace_id=234, span_id=1))
    processor.on_start(create_mock_otel_span(trace_id=345, span_id=1))

    mock_logger.warning.assert_called_once()
    warns = mock_logger.warning.call_args_list[0][0]
    assert "Creating a trace within the default" in str(warns[0])


def test_on_end():
    trace_info = create_test_trace_info(_REQUEST_ID, 0).to_v3()
    trace_manager = InMemoryTraceManager.get_instance()
    trace_manager.register_trace(_TRACE_ID, trace_info)

    otel_span = create_mock_otel_span(
        name="foo",
        trace_id=_TRACE_ID,
        span_id=1,
        parent_id=None,
        start_time=5_000_000,
        end_time=9_000_000,
    )
    span = LiveSpan(otel_span, _REQUEST_ID)
    span.set_status("OK")
    span.set_inputs({"input1": "very long input" * 100})
    span.set_outputs({"output": "very long output" * 100})

    mock_exporter = mock.MagicMock()
    mock_client = mock.MagicMock()
    mock_client.start_trace.side_effect = Exception("error")

    with mock.patch("mlflow.tracing.processor.mlflow_v2.TracingClient", return_value=mock_client):
        processor = MlflowV2SpanProcessor(span_exporter=mock_exporter)

    processor.on_end(otel_span)

    mock_exporter.export.assert_called_once_with((otel_span,))
    # Trace info should be updated according to the span attributes
    manager_trace = trace_manager.pop_trace(_TRACE_ID)
    trace_info = manager_trace.trace.info
    assert trace_info.status == TraceStatus.OK
    assert trace_info.execution_time_ms == 4
    assert trace_info.tags == {}

    # Non-root span should not be exported
    mock_exporter.reset_mock()
    child_span = create_mock_otel_span(trace_id=_TRACE_ID, span_id=2, parent_id=1)
    processor.on_end(child_span)
    mock_exporter.export.assert_not_called()


def create_test_trace_info(
    request_id,
    experiment_id="test",
    timestamp_ms=0,
    execution_time_ms=1,
    status=TraceStatus.OK,
    request_metadata=None,
    tags=None,
):
    return TraceInfoV2(
        request_id=request_id,
        experiment_id=experiment_id,
        timestamp_ms=timestamp_ms,
        execution_time_ms=execution_time_ms,
        status=status,
        request_metadata=request_metadata or {},
        tags=tags or {},
    )
