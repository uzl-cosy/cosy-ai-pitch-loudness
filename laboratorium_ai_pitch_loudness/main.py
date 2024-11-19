import argparse
import sys
import os
import json
import numpy as np
import scipy
import librosa

import laboratorium_ai_pitch_loudness.pYIN_util as pYIN
import laboratorium_ai_pitch_loudness.db_calculations as DB

# import laboratorium_ai_pitch_loudness.pYIN_util as pYIN
# import laboratorium_ai_pitch_loudness.db_calculations as DB

# Global file descriptor variable, defaulting to None
FD = None


def send_pipe_message(message):
    global FD
    if FD is not None:
        os.write(FD, message.encode("utf-8") + b"\n")
        # os.fsync(FD)


def multiple_appends(listname, *element):
    listname.extend(element)


def calc_f0_statistics(f0, voiced_flags):
    mean_f0 = np.mean(f0[voiced_flags])
    max_f0 = np.max(f0[voiced_flags])
    min_f0 = np.min(f0[voiced_flags])
    return mean_f0, max_f0, min_f0


def process_audio(in_path_audio, in_path_json, out_path_json):

    FS_TARGET = 16000

    with open(in_path_json, "r") as f:
        data = json.load(f)

    if "Start Times" not in data.keys() or "End Times" not in data.keys():
        sys.stdout.write("Start or End Times not found. Processing aborted!")
        send_pipe_message("done")
    else:
        start_times = data["Start Times"]
        end_times = data["End Times"]

        fs, audio_data = scipy.io.wavfile.read(in_path_audio)

        if audio_data.size == 0:
            return sys.stdout.write("Audio Data empty!")

        audio_data = audio_data.astype(np.float32)

        if fs != FS_TARGET:
            audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=FS_TARGET)

        f0_vals = []
        f0_statistics = []

        l_vals = []
        l_statistics = []

        for start_time, end_time in zip(start_times, end_times):
            pyin_est = pYIN.pyin(
                audio_data[int(start_time * FS_TARGET) : int(end_time * FS_TARGET)],
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=FS_TARGET,
                frame_length=int(FS_TARGET / 4),
                hop_length=512,
                fill_na=0,
            )

            if np.all(pyin_est[1] == False) == True:
                pyin_est = list(pyin_est)
                pyin_est[1] = np.invert(pyin_est[1])

            try:
                pyin_statistics = calc_f0_statistics(pyin_est[0], pyin_est[1])
                f0_tmp = pyin_est[0][pyin_est[1]]

                f0_vals.append(np.ndarray.tolist(np.around(f0_tmp, decimals=1)))
                f0_statistics.append(
                    {
                        "mean": round(pyin_statistics[0], 1),
                        "max": round(pyin_statistics[1], 1),
                        "min": round(pyin_statistics[2], 1),
                    }
                )
            except:
                pass

            db_vals = DB.compute_power_db(
                audio_data[int(start_time * FS_TARGET) : int(end_time * FS_TARGET)],
                FS_TARGET,
            )

            if not db_vals:
                db_vals = 0
                l_vals.append(db_vals)
                l_statistics.append(
                    {
                        "mean": 0,
                        "max": 0,
                        "min": 0,
                    }
                )

            else:
                l_vals.append(db_vals)

                db_statistics = DB.compute_db_statistics(db_vals)
                l_statistics.append(
                    {
                        "mean": round(db_statistics[0], 1),
                        "max": round(db_statistics[1], 1),
                        "min": round(db_statistics[2], 1),
                    }
                )

        out_dict = {
            "Pitch Values": f0_vals,
            "Pitch Statistics": f0_statistics,
            "Loudness Values": l_vals,
            "Loudness Statistics": l_statistics,
        }

        with open(out_path_json, "w") as f:
            json.dump(out_dict, f, ensure_ascii=False)

        send_pipe_message("done")


def get_paths():
    return sys.stdin.readline().strip().split(",")


def main():
    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument(
        "-f",
        "--fd",
        type=int,
        help="Optional file descriptor for pipe communication",
    )
    args = parser.parse_args()

    if args.fd:
        global FD
        FD = args.fd  # Set the global file descriptor only if provided

    send_pipe_message("ready")

    while True:
        input_path_audio, input_path_json, output_path_json = get_paths()
        process_audio(
            in_path_audio=input_path_audio,
            in_path_json=input_path_json,
            out_path_json=output_path_json,
        )


if __name__ == "__main__":
    main()
