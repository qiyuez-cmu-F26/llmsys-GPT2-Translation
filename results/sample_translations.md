# Sample Translations (epoch 19)

Greedy-decoded outputs from the final model (lr = 0.001, epoch 19, BLEU 20.6) on the first 100 sentences of the IWSLT14 De-En test split. The examples below are picked to show the range of behaviour, including failures, not just the best cases.

The text is lowercased and tokenized (spaces before punctuation) because that is how the dataset is preprocessed. HTML escapes such as `&apos;` are shown as normal characters for readability.

## Good translations

| German | Reference | Model |
|---|---|---|
| es ist interessant . | it's interesting . | it's interesting . |
| aber dies sind alle straßen in den u.s.a . | but those are all the roads in the u.s . | but these are all the streets in the u.s.a . |
| nun , was bedeutet das ? | now , what does that mean ? | now what does that mean ? |
| wir tanzen im tempel und werden zu gott . " | we dance in the temple and become god . " | we dance in the temple and god . " |
| und das problem ist nicht die technologie an sich . | and the problem is not technology itself . | and the problem is not the technology . |
| das interessante ist die metapher , welche die beziehung zwischen dem individuum und der natur definiert . | what's interesting is the metaphor that defines the relationship between the individual and the natural world . | the interesting is the metaphor that the relationship between the individual and nature . |
| eine sprache ist ausdruck des menschlichen geistes . | a language is a flash of the human spirit . | a language is expression of human mind . |
| sie sind die heimatländer von jemandem . | they're homelands of somebody . | they're the home of somebody . |

In these cases the model gets the sentence structure and most content words right. Typical slips are dropping a word ("become", "defines") or choosing a near synonym.

## Partly correct

| German | Reference | Model |
|---|---|---|
| und der entscheidende indikator dafür ist das aussterben der sprachen . | and the great indicator of that , of course , is language loss . | and the crucial indicator is the exception of the languages . |
| ich bin sehr dankbar für den empfang hier . | it's very gratifying to have this kind of reception here . | i'm very grateful for the recipient here . |
| es ist ein unabhängiges gebiet . sie kontrollieren alle mineralvorkommen . | it's an independent territory . they control all mineral resources . | it's a independent area . they all use mineral . |

The overall meaning is often recoverable, but a key word is mistranslated ("aussterben" → "exception", "empfang" → "recipient") or the second half of the sentence is thin.

## Failure cases

| German | Reference | Model | What went wrong |
|---|---|---|---|
| wir durchlaufen initiationsrituale . | we go through initiation rites . | we're going to take mass-fituits . | Rare compound word split into subwords the model never learned to compose |
| dies ist ein projekt mit dem namen photosynth , das wirklich zwei verschiedene technologien miteinander verbindet . | this is a project called photosynth , which really marries two different technologies . | this is a project with the name of fluctth , which is really two different technologies . | Product name garbled, while the rest of the sentence is fine |
| wir haben auch etwas an der ecke dieser speziellen ausgabe des the guardian gemacht . | we've also done a little something with the corner of this particular issue of the guardian . | we also have something about the corner of the corner of the corner of the city . | Repetition loop, typical of greedy decoding |
| die schwester , die diese tapfere tat beging , wurde in ein erziehungslager gebracht . | the sister who had done that brave deed was put into an education camp . | the sister that the most of the most of the most of the marriage was developed | Repetition loop; the output ends abruptly |
| diese pflanze enthält einige sehr starke tryptamine die dem serotonin , dimethyltryptamin-5 und methoxydimethyltryptamin sehr ähnlich sind . | this plant had in it some very powerful tryptamines , very close to brain serotonin , dimethyltryptamine , 5-methoxydimethyltryptamine . | *(empty output)* | Long source with rare chemical names; the source alone probably fills most of the 40-token window |

## Observations

- **Short, common sentences work well.** Sentences with everyday vocabulary and simple structure are often translated correctly or nearly so.
- **Rare words and names are the main weakness.** With a 10,000-token vocabulary trained on a small corpus and a 6.7M-parameter model, unusual words get split into fragments that the model cannot reassemble.
- **Long inputs hurt because of the context window.** Source and translation share one 40-token window, so a long German sentence leaves little or no room for the English output. This likely explains the empty output above and the many outputs that stop mid-sentence.
- **Greedy decoding can loop.** Some outputs repeat a phrase ("of the most of the most"). Beam search, a repetition penalty or a larger model would likely reduce this.
- **BLEU understates readability in places.** Several outputs are fluent and meaningful but worded differently from the single reference, which BLEU penalizes.
