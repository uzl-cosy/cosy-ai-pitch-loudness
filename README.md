# Laboratorium AI Pitch Loudness

![Python](https://img.shields.io/badge/Python-3.10.13-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Poetry](https://img.shields.io/badge/Build-Poetry-blue.svg)

**Laboratorium AI Pitch Loudness** is a Python package for analyzing pitch and loudness in audio files. It processes `.wav` audio files along with start and end times provided in a JSON file, computes pitch and loudness values and statistics for each segment, and saves the results in a JSON file. The package uses `librosa` and custom utility modules for high-quality analysis.

## Table of Contents

- [Laboratorium AI Pitch Loudness](#laboratorium-ai-pitch-loudness)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Features](#key-features)
  - [Installation and Build](#installation-and-build)
  - [Usage](#usage)
    - [CLI Usage with File Descriptors](#cli-usage-with-file-descriptors)
      - [1. Start Module](#1-start-module)
      - [2. Wait for "ready" Signal](#2-wait-for-ready-signal)
      - [3. Process Files](#3-process-files)
    - [Example Shell Script](#example-shell-script)
  - [License](#license)

## Overview

**Laboratorium AI Pitch Loudness** provides a simple way to analyze the pitch and loudness of audio recordings over specified time intervals. It supports processing of `.wav` files using start and end times, and outputs detailed pitch and loudness information.

### Key Features

- **Pitch Analysis:** Calculates pitch values and statistics (mean, max, min) for specified audio segments.
- **Loudness Analysis:** Computes loudness values and statistics (mean, max, min) in decibels (dB).
- **Custom Time Intervals:** Processes audio based on provided start and end times.
- **Flexible Configuration:** Customization of processing parameters.

## Installation and Build

This package is managed with [Poetry](https://python-poetry.org/). Follow these steps to install and build the package:

1. **Clone Repository:**

   ```bash
   git clone https://github.com/uzl-cosy/cosy-ai-pitch-loudness.git
   cd cosy-ai-pitch-loudness
   ```

2. **Install Dependencies:**

   ```bash
   poetry install
   ```

3. **Activate Virtual Environment:**

   ```bash
   poetry shell
   ```

4. **Build Package:**

   ```bash
   poetry build
   ```

   This command creates the distributable files in the `dist/` directory.

## Usage

The package runs as a persistent module through the command line interface (CLI). It enables processing of audio files and corresponding JSON files containing start and end times, and outputs the analysis to a JSON file using file descriptors. Communication occurs through a pipe, where the module sends "ready" once it's loaded and ready for processing.

### CLI Usage with File Descriptors

#### 1. Start Module

Start the Pitch Loudness module via CLI. The module signals through the file descriptor when it's ready.

```bash
python -m laboratorium_ai_pitch_loudness -f <FD>
```

**Parameters:**

- `-f, --fd`: File descriptor for pipe communication.

**Example:**

```bash
python -m laboratorium_ai_pitch_loudness -f 3
```

#### 2. Wait for "ready" Signal

After starting the module, it initializes necessary resources. Once ready, the module sends a "ready" signal through the specified file descriptor.

#### 3. Process Files

Pass the input file paths and output file path through the pipe. The module processes the files and sends a "done" signal once processing is complete.

**Input Files:**

- **Input Audio File:** The path to the `.wav` audio file to be processed.
- **Input JSON File:** Contains `"Start Times"` and `"End Times"` lists specifying the segments to analyze.

**Example:**

```bash
echo "path/to/input_audio.wav,path/to/input_times.json,path/to/output_analysis.json" >&3
```

**Description:**

- The `echo` command sends input and output file paths through file descriptor `3`.
- The module receives the paths, processes the audio data, and saves the analysis result in the output JSON file.
- Upon completion, the module sends a "done" signal through the file descriptor.

**Complete Flow:**

1. **Start the Pitch Loudness Module:**

   ```bash
   python -m laboratorium_ai_pitch_loudness -f 3
   ```

2. **Send File Paths for Processing:**

   ```bash
   echo "path/to/input_audio.wav,path/to/input_times.json,path/to/output_analysis.json" >&3
   ```

3. **Wait for "done" Signal:**

   After sending the file paths, wait for the module to process the files. It will send a "done" signal when processing is complete.

4. **Repeat Step 2 for Additional Files:**

   You can process additional files by repeating the file path input:

   ```bash
   echo "path/to/another_input_audio.wav,path/to/another_input_times.json,path/to/another_output_analysis.json" >&3
   ```

### Example Shell Script

Here's an example of using the Pitch Loudness package in a shell script:

```bash
#!/bin/bash

# Open a file descriptor (e.g., 3) for pipe communication

exec 3<>/dev/null

# Start the Pitch Loudness module in background and connect the file descriptor

python -m laboratorium_ai_pitch_loudness -f 3 &

# Store module's PID for later termination if needed

PL_PID=$!

# Wait for "ready" signal

read -u 3 signal
if [ "$signal" = "ready" ]; then
echo "Module is ready for processing."

      # Send input and output paths
      echo "path/to/input_audio.wav,path/to/input_times.json,path/to/output_analysis.json" >&3

      # Wait for "done" signal
      read -u 3 signal_done
      if [ "$signal_done" = "done" ]; then
            echo "Processing complete."
      fi

      # Additional processing can be added here
      echo "path/to/another_input_audio.wav,path/to/another_input_times.json,path/to/another_output_analysis.json" >&3

      # Wait for "done" signal again
      read -u 3 signal_done
      if [ "$signal_done" = "done" ]; then
            echo "Additional processing complete."
      fi

fi

# Close the file descriptor

exec 3>&-
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
