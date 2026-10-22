import pytest

from resilens.cli import main


def test_inspect_valid_video(make_video, capsys) -> None:
    main(["inspect", str(make_video(frames=5, fps=10.0))])

    output = capsys.readouterr().out
    assert "5 frames" in output
    assert "Replayed 5 frames" in output


def test_inspect_without_frames_flag_prints_no_table(make_video, capsys) -> None:
    main(["inspect", str(make_video(frames=5))])

    assert "read_ms" not in capsys.readouterr().out


def test_inspect_frames_prints_row_per_frame(make_video, capsys) -> None:
    main(["inspect", str(make_video(frames=5, fps=10.0)), "--frames"])

    lines = capsys.readouterr().out.splitlines()
    header = next(i for i, line in enumerate(lines) if "read_ms" in line)
    rows = [line.split() for line in lines[header + 1 : header + 6]]
    assert [int(row[0]) for row in rows] == [0, 1, 2, 3, 4]
    assert [float(row[1]) for row in rows] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert all(float(row[5]) > 0 for row in rows)  # RAM in MB
    assert any(line.startswith("Read time per frame:") and "p95" in line for line in lines)


def test_inspect_corrupted_video_exits_with_message(tmp_path, capsys) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"broken")

    with pytest.raises(SystemExit) as exit_info:
        main(["inspect", str(path)])

    assert exit_info.value.code == 1
    assert "Invalid video" in capsys.readouterr().err


def test_inspect_respects_limits(make_video, capsys) -> None:
    path = make_video(width=64, height=48)

    with pytest.raises(SystemExit):
        main(["inspect", str(path), "--max-resolution", "32", "32"])

    assert "exceeds the maximum" in capsys.readouterr().err
