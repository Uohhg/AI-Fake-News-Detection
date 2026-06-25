
# Nigerian Language and Context Checker
# Detects Nigerian Pidgin, Yoruba, Hausa
# and common Nigerian fake news patterns

NIGERIAN_PIDGIN_FAKE_PHRASES = [
    'dem dey hide', 'government dey lie', 'dem no want us know',
    'share before dem delete', 'na lie dem dey tell',
    'dem don ban am', 'this one na real talk', 'e don cast',
    'na una dem dey use', 'dem dey use us', 'open your eyes',
    'dem dey fear this post', 'before they remove am',
    'share reach your people', 'pass am to everybody'
]

NIGERIAN_PIDGIN_REAL_PHRASES = [
    'dem say', 'e be like say', 'as e stand',
    'dem confirm say', 'na so e be'
]

YORUBA_FAKE_INDICATORS = [
    'eyin eniyan', 'e jo share', 'won fe pa wa', 'ijoba n pa wa',
    'eyin omo oodua', 'e jo gba', 'ki enu re ma pa'
]

HAUSA_FAKE_INDICATORS = [
    'gwamnati na boye', 'share kafin share', 'su na boye gaskiya',
    'ku yada wannan', 'kafin a share'
]

NIGERIAN_TRUSTED_OUTLETS = [
    'punchng.com', 'channelstv.com', 'vanguardngr.com',
    'premiumtimesng.com', 'thecable.ng', 'thisdaylive.com',
    'dailytrust.com', 'tribuneonlineng.com', 'ngrguardiannews.com',
    'businessdayng.com', 'saharareporters.com'
]

NIGERIAN_FAKE_OUTLETS = [
    'naijagists.com', 'lindaikejisblog.com', 'instablog9ja.com'
]

NIGERIAN_FAKE_NEWS_TOPICS = [
    'tinubu resign', 'nigeria break up', 'naira crash',
    'military coup', 'president dead', 'nigeria end',
    'biafra restoration', 'nigeria breaking',
    'buhari secret', 'tinubu secret', 'dollar rate crash'
]

def check_nigerian_context(article_text):
    text_lower = article_text.lower()
    result = {
        'is_nigerian_content': False,
        'language_detected': 'English',
        'nigerian_fake_phrases': [],
        'nigerian_real_phrases': [],
        'nigerian_fake_topics': [],
        'nigerian_score': 50,
        'explanation': []
    }

    # Check for Nigerian Pidgin fake phrases
    pidgin_fake_found = [
        p for p in NIGERIAN_PIDGIN_FAKE_PHRASES
        if p in text_lower
    ]
    if pidgin_fake_found:
        result['is_nigerian_content'] = True
        result['language_detected'] = 'Nigerian Pidgin English'
        result['nigerian_fake_phrases'] = pidgin_fake_found
        result['nigerian_score'] -= len(pidgin_fake_found) * 15
        result['explanation'].append(
            "Contains " + str(len(pidgin_fake_found)) +
            " Nigerian Pidgin fake news phrase(s)"
        )

    # Check for Yoruba fake indicators
    yoruba_found = [
        p for p in YORUBA_FAKE_INDICATORS
        if p in text_lower
    ]
    if yoruba_found:
        result['is_nigerian_content'] = True
        result['language_detected'] = 'Yoruba'
        result['nigerian_score'] -= len(yoruba_found) * 15
        result['explanation'].append(
            "Contains " + str(len(yoruba_found)) +
            " Yoruba fake news indicator(s)"
        )

    # Check for Hausa fake indicators
    hausa_found = [
        p for p in HAUSA_FAKE_INDICATORS
        if p in text_lower
    ]
    if hausa_found:
        result['is_nigerian_content'] = True
        result['language_detected'] = 'Hausa'
        result['nigerian_score'] -= len(hausa_found) * 15
        result['explanation'].append(
            "Contains " + str(len(hausa_found)) +
            " Hausa fake news indicator(s)"
        )

    # Check for Nigerian fake news topics
    topics_found = [
        t for t in NIGERIAN_FAKE_NEWS_TOPICS
        if t in text_lower
    ]
    if topics_found:
        result['is_nigerian_content'] = True
        result['nigerian_fake_topics'] = topics_found
        result['nigerian_score'] -= len(topics_found) * 10
        result['explanation'].append(
            "Contains " + str(len(topics_found)) +
            " Nigerian fake news topic(s): " +
            ', '.join(topics_found[:3])
        )

    # Check for Nigerian real phrases
    real_found = [
        p for p in NIGERIAN_PIDGIN_REAL_PHRASES
        if p in text_lower
    ]
    if real_found:
        result['is_nigerian_content'] = True
        result['nigerian_real_phrases'] = real_found
        result['nigerian_score'] += len(real_found) * 10
        result['explanation'].append(
            "Contains " + str(len(real_found)) +
            " credible Nigerian reporting phrase(s)"
        )

    # Cap the score
    result['nigerian_score'] = max(0, min(100, result['nigerian_score']))

    if not result['is_nigerian_content']:
        result['explanation'].append("No specific Nigerian language patterns detected")

    return result
