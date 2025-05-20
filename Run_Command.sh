#!/bin/bash

# Usage: ./run_tracker.sh <TrackerName> [test|train]
# Place and run this script inside the ALL_Trackers directory

if [ -z "$1" ]; then
  echo "Usage: $0 <TrackerName> [test|train]"
  exit 1
fi

TRACKER_NAME=$1
MODE=${2:-test}  # Default to 'test' if not specified

TRACKER_DIR="./$TRACKER_NAME"

if [ "$MODE" == "test" ]; then
  RUN_CMD_FILE="$TRACKER_DIR/Run_test.txt"
elif [ "$MODE" == "train" ]; then
  RUN_CMD_FILE="$TRACKER_DIR/Run_train.txt"
else
  echo "Error: Mode must be 'test' or 'train'"
  exit 1
fi

# Check if the tracker directory exists
if [ ! -d "$TRACKER_DIR" ]; then
  echo "Error: Tracker folder '$TRACKER_NAME' does not exist."
  exit 1
fi

# Check if the command file exists
if [ ! -f "$RUN_CMD_FILE" ]; then
  echo "Error: $RUN_CMD_FILE not found in '$TRACKER_NAME'."
  exit 1
fi

# Read the command
CMD=$(<"$RUN_CMD_FILE")
echo "Running ($MODE) in $TRACKER_NAME: $CMD"

# Run the command in a subshell inside the tracker's directory
(
  cd "$TRACKER_DIR" || exit 1
  exec $CMD
)

