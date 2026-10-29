# Defense day checklist (SLT)

## The day before

- [ ] The program is on **two** carriers: the laptop that will be used and a USB stick
      (the whole `SLT` folder, not only `SLT.exe`), plus the 2-3 minute demo video.
- [ ] Run `SLT.exe` on the laptop that will be used at the defense: all three modes and
      the **?** button → *Demo 1* work.
- [ ] `progress.json` is next to `SLT.exe` if Learn mode should show the learned signs.
- [ ] Practise the demo signs (the reliable ones, 100% success in tests):
      **hello, please, happy, hat, shower, airplane, high, find, sick, balloon**.
      Avoid *mom* and *because* live (weaker in tests).
- [ ] The team has `reports/team_kit/` (README, METRICS, figures, screenshots).

## 30 minutes before

- [ ] Laptop on the charger; Windows power mode "Best performance"; close Chrome and
      other heavy programs (the frame rate drops when the CPU is busy or hot).
- [ ] Light **in front** of the signer, not behind. A plain background helps.
- [ ] Sit so that the head and **both shoulders** are in the picture, not closer: the
      raised hand must stay inside the frame.
- [ ] Start `SLT.exe` and wait until the skeleton appears (about 5-10 s after the start).
- [ ] The fps chip on the video shows **15 or more** (12 is still usable).
- [ ] Interface language (HY / RU / EN, top right) as planned for the audience.
- [ ] Sound on and loud enough: show one sign in Single word and press *Speak*.
- [ ] An external camera instead of the built-in one: Settings → camera number (0, 1, ...).
- [ ] One test sign in each mode.

## During the demo (suggested order)

1. **Single word**: press Space, wait for 3-2-1, show *hello*. Then one more (*airplane*).
2. **Sentence**: tick "Record each word with the button" (the reliable variant) and show
   *hello please happy hat shower*, one word per Space. If time allows, untick it and show the
   automatic variant: lower the hand fully between signs.
3. **Learn**: pick a sign, play the reference, press *Try*, show the side-by-side replay.
4. Numbers: one slide with `figures/accuracy_progression.png`; say "on people the model
   never saw" and give top-1 and top-3 together.

## If something goes wrong

| Problem | What to do |
|---|---|
| A sign is not recognised | Repeat once, a bit slower, hand fully in the frame. Then move on. |
| Wrong word in a sentence | Tap the right word among the alternatives under the last word. |
| Automatic Sentence misses or merges words | Switch to "Record each word with the button". |
| "Camera not found" / black video | Close other programs that use the camera (Zoom, Teams, browser), Settings → another camera number, or restart `SLT.exe`. |
| No camera at all | **?** button → *Demo 1/2/3*: the recorded sentences play through the whole program. |
| The program does not start or freezes | Show the demo video from the USB stick. Details for later: `logs\app.log` next to `SLT.exe`. |
| Very low fps (below 12) | Plug in the charger, close other programs, wait a minute. |

## What to say about the limits (if asked)

- It recognises **isolated signs** (250 everyday signs), not grammatical sign-language
  sentences; the Sentence mode is a row of single signs.
- 62.7% top-1 / 76.9% top-3 on 4 people the model never saw (65.1% with the app pipeline);
  it varies by person (57-71%).
- Everything runs offline on the laptop; the camera picture is never saved.
