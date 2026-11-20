# Privacy

StyleSense AI handles face photographs, which are personal and potentially biometric data. This document states
how they are handled.

## App users

- Photos are processed **in memory** and are **not written to disk** by default.
- Only numeric features (face-shape probabilities, landmark ratios, hair-attribute probabilities) and ratings are
  stored, linked to a random session ID. No names, emails or images.
- Every generated image carries a visible "AI-generated" label.
- The system edits only hair and beard regions. It never changes facial features, age, body shape or skin colour.

## Contributors (fairness set) and user-study participants

- Photos are collected only with written consent ([docs/consent_form.md](docs/consent_form.md)) that states purpose,
  retention period and the right to withdraw.
- Consent to train, to evaluate and to appear in published material are separate choices.
- Photos are stored privately under random IDs; consent records are kept separately.
- Skin-tone labels (Monk Skin Tone scale) are used only for aggregate fairness evaluation and are never shown to
  users.
- Photos are deleted at the end of the agreed retention period or on request (within 7 days), and the deletion is
  logged.

## What is never done

- No images scraped from social media or the web.
- No participant images in the public repository, demo video or write-up without separate opt-in.
- No sale or sharing of data with third parties.
