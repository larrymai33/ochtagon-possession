# Organizer rules clarification

Recorded from the organizer email supplied by the user on **2026-11-03**.
The email's transmission date was not supplied. The title was
**“Clarification of the rules”**. This record is not a new organizer ruling.

The email states that its rules take precedence over previous documentation,
emails and answers to individual questions. Apply it wherever older text in this
research repository conflicts, including the earlier permission for unsupervised
learning on the provided game. The organizer's substantive instructions follow.

## What is not allowed

> - Checking your output against the video.
> - Hand labels, human-produced data or any other data created manually on the provided game.
> - Uploading the provided data to AI services.
> - Training, fitting or calibrating your algorithm on the ground-truth possession data or on the provided video.

## What is allowed

> - Running automatic checks using only the provided data.
> - Inspecting the provided data and the video to better understand what counts as possession.
> - Using as input the video, the automatic tracking data, the smart tagging or any other generated data, as long as your algorithm was not trained on the ground truth or on the provided video.
> - Providing files of intermediate steps.
> - Using any dataset you have the rights to use. Please check the license of any dataset before using it.

## Training and tuning

> If you need to train (or retrain), you MUST use only external data. If your algorithm has parameters that need to be tuned, you MUST tune them on external data, never on the ground-truth possession data or on the provided video.

The email invites participants to contact the organizers with questions. This
record does not send a message or establish a decision about this team's history.

## Repository enforcement

The following are repository implementation decisions applying the clarification:

- The historical target hand-labelling, label-assembly, output/video overlay,
  target-GT calibration, calibrated decoder, video colour fitting and legacy
  feature-cache entry points are disabled at import and CLI entry. The seven
  modules are `analysis/label_tool.py`, `analysis/own_gt.py`,
  `analysis/overlay.py`, `analysis/gt_vs_tracking.py`, `solution/decode.py`,
  `solution/team_colour.py` and `solution/features.py`.
- Each guard runs before dependencies or data access. There is no override flag.
  The previous source remains below the guard for historical audit; commands or
  permission statements in that preserved source are not current instructions.
- External training and inference modules remain available. The official starter
  evaluation tools remain available for automatic pipeline checks. A target check
  must never become training, parameter selection, calibration or output/video
  comparison; the current rebuild design does not authorize a new target check.
- The preserved release, frozen source archives and historical records are not
  rewritten by this repair. Existing local outputs are not evidence of permission
  to reuse them. Future training and tuning use external data only.

These changes prevent execution through the listed entry points. They do not
erase earlier target-GT tuning, video fitting, generated comparison clips or the
documented public-GT disclosure, and they do not establish competition eligibility.
See [the historical ledger](METHODOLOGY.md) and
[the current rebuild design](docs/superpowers/specs/2026-11-03-external-only-rebuild-design.md).
