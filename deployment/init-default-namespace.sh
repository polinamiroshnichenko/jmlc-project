#!/bin/sh

echo "Waiting for Temporal server at ${TEMPORAL_ADDRESS}..."
until temporal operator namespace list > /dev/null 2>&1; do
  sleep 2
done

temporal operator namespace describe -n "${DEFAULT_NAMESPACE}" > /dev/null 2>&1 \
  || temporal operator namespace create -n "${DEFAULT_NAMESPACE}" > /dev/null 2>&1 \
  || true

echo "Namespace '${DEFAULT_NAMESPACE}' is ready"
