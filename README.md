# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/MahmudulOG/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/MahmudulOG/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour
by hour" bounty
([gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)).
Given everything recorded up to the end of hour 1 of Grand Theft Auto VI's
release day (2026-11-19, UTC) — that hour's Twitch viewer count included — it
writes five quantiles (5%, 25%, 50%, 75%, 95%) of the viewer count for each
of hours 2 to 24.

How it works: every case file carries dozens of past big game launches with
their full hourly viewer curves. A launch day's shape is much more alike in
relative terms than in absolute ones, so `main.py` pools the comparables'
log-ratios to their own hour-1 counts, per hour, weighting each comparable by
how close its hour-1 audience was to this case's hour-1 print. The weighted
mean and spread of those log-ratios give a per-hour curve around the case's
own first hour. The weighting and spread constants were tuned by
leave-one-out cross-validation over the comparables themselves, scored with
the contest's exact pinball loss on the log scale: on launches shaped like
the practice cases the tuned model scores about 0.27, against the house
persistence baseline's 0.45 on the same cross-validation.

- `main.py` — the entry: Python 3.13, standard library only, no network,
  deterministic. Reads every `<id>.in` in the input directory and writes
  `<id>.out` per case.

MIT licensed — see `LICENSE`.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=eg9w3128&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
