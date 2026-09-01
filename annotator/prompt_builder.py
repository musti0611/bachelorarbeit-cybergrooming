"""Build the few-shot system prompt for the behavior annotator."""

from parse_goldstandard import LABELS

LABEL_DESCRIPTIONS = {
    "NEUTRAL": (
        "Everyday conversation with no grooming intent. Could appear in any normal conversation.",
        "No pet names, no affective markers, no sexual content, no secrecy, no meeting plans.",
    ),
    "VERTRAUENSAUFBAU": (
        "Building emotional trust or attachment toward the victim.",
        "Pet names (sweetie/honey/baby/cutie), compliments on appearance or personality, "
        "love declarations, expressing missing the victim, emphasizing exclusivity ('you're the only one'), "
        "promises of care or protection. "
        "Also: affective emoticons (:-*, >:D<, xoxo, hugs/kisses symbols), "
        "expressing devotion or persistence ('I waited for you', 'I kept trying to reach you'), "
        "claiming to be trustworthy or safe ('I respect you', 'I would never hurt you', 'I don't take advantage of people'), "
        "expressing amazement or gratitude at the victim's attention ('I can't believe you're talking to me'). "
        "RULE: Any pet name, affective emoticon, or expression of devotion/trust → VERTRAUENSAUFBAU, even if the message looks otherwise harmless.",
    ),
    "INFORMATIONSGEWINNUNG": (
        "Gathering personal information to assess the victim's vulnerability, location, and accessibility.",
        "Questions about address/city/location, parent's schedule or whereabouts, daily routine, "
        "siblings, social network, sexual history as risk assessment ('have you done X before?'). "
        "RULE: Sexual question aimed at assessing victim experience/vulnerability → INFORMATIONSGEWINNUNG, not SEXUALISIERUNG.",
    ),
    "GEHEIMHALTUNG/ISOLATION": (
        "Keeping the relationship or actions secret; avoiding detection; isolating victim from others.",
        "Instructions to delete chats/messages, asking if anyone can observe them, "
        "planning to avoid being seen, asking victim to lie to parents or friends, "
        "emphasizing that the relationship must stay secret.",
    ),
    "SEXUALISIERUNG": (
        "Introducing, describing, normalizing, or requesting sexual content.",
        "Explicit sexual language or descriptions, requesting sexual acts or images, "
        "sexual evaluation of body parts, describing own sexual arousal, sexual roleplay. "
        "RULE: Sexual content without a clear information-gathering purpose → SEXUALISIERUNG.",
    ),
    "KONTROLLE/NOETIGUNG": (
        "Psychological pressure, emotional manipulation, threats, or coercion to control the victim's behavior.",
        "Emotional blackmail / guilt-tripping ('you're making me feel bad', 'nobody loves me'), "
        "passive aggression ('I guess you don't want to talk anymore', 'fine, goodbye'), explicit threats, "
        "direct commands or instructions about the victim's behavior or appearance ('wear a skirt for me', "
        "'be ready for me', 'answer the door like that'), degradation of the victim ('lil slut'). "
        "RULE: A direct command or instruction about the victim's behavior, appearance, or actions = KONTROLLE/NOETIGUNG, "
        "even if the commanded action is sexual in nature. "
        "RULE: Guilt-tripping or self-pity used to pressure the victim = KONTROLLE/NOETIGUNG, even if it sounds sad. "
        "RULE: If coercion or manipulation dominates → KONTROLLE/NOETIGUNG, even if sexual content is also present.",
    ),
    "OFFLINE-ESKALATION": (
        "Planning, arranging, or preparing for physical (real-world) contact.",
        "Arranging meetings (time/place), exchanging phone numbers, discussing travel or directions, "
        "confirming or modifying planned meetings. "
        "RULE: Address/location question in context of travel or meeting planning → OFFLINE-ESKALATION, not INFORMATIONSGEWINNUNG.",
    ),
}

KEY_RULES = """KEY DECISION RULES:
- Pet names, affective emoticons (:-*, >:D<, xoxo), or expressions of devotion → VERTRAUENSAUFBAU (even if rest is neutral)
- Claiming to be trustworthy or safe ('I respect you', 'I would never hurt you') → VERTRAUENSAUFBAU
- Expressing devotion or persistence ('I waited for you', 'I kept trying to reach you') → VERTRAUENSAUFBAU
- Direct command about victim's behavior or appearance ('wear this', 'be ready for me') → KONTROLLE/NOETIGUNG (even if sexual)
- Guilt-tripping or self-pity used to pressure ('nobody loves me', 'you don't care') → KONTROLLE/NOETIGUNG
- Degrading language toward victim → KONTROLLE/NOETIGUNG (not SEXUALISIERUNG)
- Sexual question about victim's past experience (risk assessment) → INFORMATIONSGEWINNUNG
- Sexual content without information purpose → SEXUALISIERUNG
- Secrecy instruction without explicit coercion → GEHEIMHALTUNG/ISOLATION
- Secrecy with explicit threat or coercion → KONTROLLE/NOETIGUNG
- Location/address question without meeting context → INFORMATIONSGEWINNUNG
- Location/address question for travel or meeting planning → OFFLINE-ESKALATION"""


def build_system_prompt(data: dict, n_fewshot: int = 3) -> str:
    """
    Build system prompt with label definitions and few-shot examples.
    Uses first n_fewshot examples per label from data.
    """
    lines = [
        "You are an expert annotator for cybergrooming research.",
        "",
        "Classify the chat message into exactly ONE of these 7 labels based on its observable "
        "communicative behavior — the surface content of the message itself, regardless of who "
        "wrote it or what their underlying intent may be.",
        "Reply with ONLY the label name. Nothing else. No explanation.",
        "",
        "=== LABELS ===",
        "",
    ]

    for label in LABELS:
        definition, indicators = LABEL_DESCRIPTIONS[label]
        lines.append(label)
        lines.append(f"Definition: {definition}")
        lines.append(f"Indicators: {indicators}")
        lines.append("")

    lines.append(KEY_RULES)
    lines.append("")
    lines.append("=== FEW-SHOT EXAMPLES ===")
    lines.append("")

    for label in LABELS:
        examples = data[label][:n_fewshot]
        for ex in examples:
            lines.append(f'Text: "{ex["text"]}" → {label}')
        lines.append("")

    lines.append("=== VALID OUTPUT VALUES (use exactly as written) ===")
    lines.append(", ".join(LABELS))

    return "\n".join(lines)
