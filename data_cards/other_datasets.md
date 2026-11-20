# Data cards: Figaro-1k, FairFace, FFHQ

## Figaro-1k
| Field | Value |
|---|---|
| Size | 1,050 images, 7 hairstyle classes x 150, with hair masks |
| Licence | Research use |
| Used for | Curl-pattern head of the hair-attribute classifier (WBS 4.2); extra hair masks |
| Limitations | Small; mixed angles and backgrounds; limited beard content |

## FairFace
| Field | Value |
|---|---|
| Source | Karkkainen & Joo, WACV 2021 - https://github.com/dchen236/FairFace |
| Licence | **CC BY 4.0** - attribution required |
| Used for | Additional South Asian ("Indian" group) and darker-skin faces for training interventions and evaluation (WBS 6) |
| Notes | Race labels are **not** used as a skin-tone proxy; selected images are re-annotated on the MST scale |

## FFHQ (subset)
| Field | Value |
|---|---|
| Source | NVIDIA - https://github.com/NVlabs/ffhq-dataset |
| Size used | ~5,000 of 70,000 |
| Licence | **CC BY-NC-SA 4.0** (dataset); individual Flickr images under various licences |
| Used for | Generative testing and KID reference set (WBS 5.4) |
| Notes | Non-commercial; attribution; derivatives under the same licence |
