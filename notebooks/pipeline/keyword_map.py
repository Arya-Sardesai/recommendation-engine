"""
keyword_map.py  --  TMDB keyword -> curated taxonomy mapping (movies + TV).

WHAT THIS IS (and is NOT)
-------------------------
A DETERMINISTIC LOOKUP TABLE, not a model. If TMDB says a title carries the
keyword "lgbt", it gets the curated tag "queer-rep" at 0.7. No LLM, no API
cost, no inference. Contrast tag_data.py, which sends book descriptions to
Haiku and asks it to *read* them.

The three tag sources are complementary and reach different things:
    keywords -> what a title is ABOUT   (crime, queer-rep, grief, heist, war)
    genome   -> how it FEELS to watch   (atmospheric, slow-burn, visually-stunning)
    LLM      -> either, inferred from prose

MEASURED CONSEQUENCE: keyword-derived tags are strong on theme, weak on
experience/pace/craft. Because TV has NO genome equivalent, TV is permanently
strong on theme and weak on experience. Do NOT assume symmetry between
modalities when building anything cross-modal on top of these tags.

SCORING -- TWO TIERS, deliberately not three
--------------------------------------------
    PRIMARY    0.7   the keyword essentially IS the tag  (heist -> heist)
    SUPPORTING 0.5   strongly implies it                 (bank robbery -> heist)

Score = 0.7 if ANY primary matches, else 0.5 if any supporting matches. NOT
additive: three supporting keywords are the same evidence stated three times,
not better evidence.

Three tiers were rejected. Keyword presence is boolean; 0.8/0.6/0.4 fabricates
precision the data doesn't contain, and ranking multiplies by these scores so
invented gradations become real ordering. Anything justifiable only at a
hypothetical "weak 0.4" is EXCLUDED instead. Precision discipline is structural
rather than a fudgeable number.

VALIDATION
----------
Every source string was checked to occur in data/tmdb_raw/kw_freq_{movies,tv}.csv.
The first draft had 34 invented strings (>20%) -- 'practical effects',
'ensemble cast', 'cinematography', 'fast paced' are simply not TMDB keywords.
If you ADD a source, re-validate it or it silently contributes nothing.

REVISION 2 -- corrections from the first precision audit
--------------------------------------------------------
Sources deleted for demonstrated sweep-in (audited sample -> verdict):
    cyberpunk         <- dystopia            Interstellar, Star Wars, Hunger Games
    dark-academia     <- university, college  Toy Story 3, Monsters University
    whimsical         <- surrealism          The Shining
    horror            <- monster             Monsters Inc., Pacific Rim
    crime             <- murder              The Lion King
    romance           <- romantic            Whiplash, Ratatouille
    stylized-violence <- martial arts        Big Hero 6
    forbidden-love    <- infidelity, adultery  Gone Girl (infidelity != forbidden love)
    psychological     <- obsession           all three LOTR films
    mind-bending      <- dream               The Mask, Narnia
    satire            <- social commentary   Despicable Me, Ted
    meditative        <- meditation          Headspace Guide to Meditation
    identity          <- amnesia             The Hangover, Finding Dory
    heist             <- theft               Zootopia
    political-intrigue <- conspiracy         Inception, Zootopia (kept in thriller)
    man-vs-nature     <- survival, shipwreck  returned a sample list byte-identical
                                              to `survival`; now wilderness +
                                              natural-disaster only

Tags REMOVED from this map entirely (still declared in the taxonomy, where
genome and the book LLM continue to populate them):
    hubris             20 items, every sample wrong (The Heat, Just Friends)
    fast-paced         Interstellar and 1917 are the opposite; no primary source
    visually-stunning  1 item; genome covers it well for movies

REVISION 3 -- corrections from the second audit (new tags)
-----------------------------------------------------------
    sports         <- martial arts    The Matrix, Kill Bill, John Wick, Kung Fu
                                      Panda, Naruto. Martial-arts films are not
                                      sports films -> promoted to its OWN tag.
    true-crime     <- serial killer   Se7en, Silence of the Lambs, Dexter, Death
                                      Note: all FICTION about serial killers.
                                      Only the `true crime` primary survives.
    slice-of-life  <- school life     My Hero Academia, Jujutsu Kaisen, Haikyu --
                                      shounen action set in schools, the opposite
                                      of slice-of-life.
    dark-academia  <- secret society  Spectre, John Wick 3. SPECTRE is literally
                                      a secret society. Demoted to supporting.
    cyberpunk      <- android         Star Wars, The Empire Strikes Back (droids).

DELIBERATELY KEPT despite imprecision:
    biographical <- based on true story   catches Titanic, Dunkirk, Saving
    Private Ryan, which are true-story but not biographies. Judged an acceptable
    stretch at 0.5 supporting rather than losing 1,629 instances of real signal.

KNOWN-WEAK BUT KEPT -- low recall, high precision. The audit showed these are
correct where they fire, and they are exactly the "experience" tags the
folksonomy otherwise cannot express:
    slow-burn 18    Jeanne Dielman, The Turin Horse, Drive My Car
    atmospheric 26  Aftersun, Skinamarink, Picnic at Hanging Rock
    long-take 9     Victoria, Fish & Cat
    twist-ending 37, meditative 50, found-family 68, urban-fantasy 6

Trailing comments show approximate keyword mass (movies / TV) -- an upper bound
on how many titles can receive the tag, not the count.
"""

# tag -> (PRIMARY sources, SUPPORTING sources).  All lowercase, exact match.
KEYWORD_MAP = {
    # ---- source medium: the books<->screen bridge (NEW, largest block) ----
    "book-adaptation": (['based on novel or book'], ['based on young adult novel', 'based on play or musical', 'based on light novel']),  # ~5,100m ~1,520t
    "comic-adaptation": (['based on comic'], ['superhero']),  # ~1,220m ~710t
    "manga-adaptation": (['based on manga'], []),  # ~641m ~1,297t
    "game-adaptation": (['based on video game'], []),  # ~150m ~215t

    # ---- sub-genres the TMDB genre column cannot express (NEW) ----
    "superhero": (['superhero'], ['super power', 'secret identity']),  # ~1,060m ~730t
    "noir": (['film noir'], ['neo-noir']),  # ~546m
    "sports": (['sports'], ['boxing', 'football', 'baseball', 'basketball']),  # ~1,045m ~266t
    "martial-arts": (['martial arts', 'kung fu'], ['samurai', 'ninja', 'wuxia', 'karate', 'shaolin']),  # ~1,423m ~252t
    "biographical": (['biography'], ['based on true story']),  # ~1,633m
    "true-crime": (['true crime'], []),  # ~243t -- fiction ABOUT serial killers is not true crime
    "supernatural": (['supernatural'], ['ghost', 'vampire', 'demon', 'possession']),  # ~534m ~600t

    # ---- anime vocabulary (NEW) ----
    "shounen": (['shounen'], []),  # ~393t
    "seinen": (['seinen'], []),  # ~295t
    "mecha": (['mecha'], []),  # ~294t
    "slice-of-life": (['slice of life'], []),  # ~652t

    # ---- TV format (NEW) ----
    "miniseries": (['miniseries'], []),  # ~1,209t
    "sitcom": (['sitcom'], []),  # ~829t
    "telenovela": (['telenovela'], []),  # ~282t

    # ---- movie format (NEW) ----
    "short-film": (['short film'], []),  # ~2,239m

    # ---- speculative / genre ----
    "science-fiction": (['science fiction'], ['futuristic', 'alien', 'spacecraft', 'space travel']),
    "dystopian": (['dystopia'], ['totalitarianism', 'authoritarianism', 'surveillance', 'oppression']),
    "space-opera": (['space opera'], ['spacecraft', 'space travel', 'alien planet', 'galaxy']),
    "cyberpunk": (['cyberpunk'], ['virtual reality', 'cyborg', 'hacker']),
    "climate-fiction": (['climate change', 'global warming'], ['environmentalism', 'ecology', 'pollution', 'natural disaster']),
    "time-travel": (['time travel'], ['time machine', 'alternate timeline', 'time paradox']),
    "time-loop": (['time loop'], ['repeating day']),
    "post-apocalyptic": (['post-apocalyptic future', 'apocalypse'], ['wasteland', 'survivalism', 'end of the world', 'zombie apocalypse']),
    "fantasy": (['fantasy', 'sword and sorcery'], ['magic', 'mythology', 'fairy tale', 'supernatural power']),
    "high-fantasy": (['sword and sorcery', 'epic fantasy'], ['dragon', 'wizard', 'elves', 'medieval fantasy', 'quest']),
    "urban-fantasy": (['urban fantasy'], []),
    "historical": (['period drama', 'historical event', 'history'], ['19th century', '18th century', 'medieval', 'ancient rome']),
    "mystery": (['mystery', 'whodunit'], ['detective', 'investigation', 'private detective', 'murder mystery']),
    "thriller": (['psychological thriller', 'suspense'], ['cat and mouse', 'manhunt', 'conspiracy']),
    "horror": (['horror'], ['slasher', 'haunting', 'supernatural horror', 'zombie', 'possession']),
    "romance": (['romance', 'love story'], ['falling in love', 'courtship']),
    "war": (['war', 'world war ii', 'world war i'], ['soldier', 'battlefield', 'military', 'combat', 'resistance']),
    "crime": (['crime', 'organized crime'], ['gangster', 'mafia', 'criminal', 'drug trade']),
    "coming-of-age": (['coming of age'], ['adolescence', 'teenager', 'growing up', 'first love']),
    "satire": (['satire', 'satirical'], ['parody', 'spoof', 'mockumentary']),
    "heist": (['heist', 'bank robbery'], ['robbery', 'con artist', 'burglary']),
    "dark-academia": (['boarding school'], ['professor', 'secret society']),

    # ---- mood ----
    "dark": (['dark comedy', 'dark fantasy'], ['nihilism', 'bleak', 'cynicism', 'moral ambiguity']),
    "hopeful": (['hope'], ['inspirational', 'second chance']),
    "melancholic": (['melancholy'], ['loneliness', 'nostalgic', 'bittersweet', 'gloomy', 'wistful']),
    "humorous": (['comedy', 'humor'], ['slapstick', 'farce', 'screwball comedy', 'buddy comedy']),
    "bleak": (['bleak'], ['despair', 'nihilism', 'depressing', 'hopelessness']),
    "feel-good": (['feel good', 'feel-good'], ['heartwarming', 'uplifting', 'lighthearted']),
    "tense": (['suspenseful', 'tense'], ['paranoia', 'claustrophobic', 'cat and mouse']),
    "whimsical": (['whimsical'], ['eccentric', 'absurdism']),
    "eerie": (['eerie', 'creepy'], ['unsettling', 'haunting']),
    "atmospheric": (['atmospheric'], ['moody', 'dreamlike']),
    "witty": (['witty', 'screwball comedy'], ['wordplay']),
    "heart-wrenching": (['tragedy'], ['tragic', 'heartbreak', 'loss of loved one', 'terminal illness']),
    "meditative": (['slow cinema'], ['introspection', 'solitude']),
    "thought-provoking": (['philosophical', 'philosophy'], ['existentialism', 'moral dilemma']),
    "emotionally-heavy": (['trauma'], ['depression', 'grief', 'abuse', 'addiction']),

    # ---- experience / pace / structure ----
    "slow-burn": (['slow burn'], ['slow cinema']),
    "mind-bending": (['surrealism'], ['psychedelic']),
    "tearjerker": (['tearjerker'], ['terminal illness', 'heartbreak', 'death of a child']),
    "twist-ending": (['twist ending', 'plot twist'], ['surprise ending', 'unreliable narrator']),
    "non-linear": (['nonlinear timeline'], ['flashback', 'multiple storylines', 'anthology']),

    # ---- relationships / conflict / theme ----
    "found-family": (['found family'], ['chosen family', 'unlikely friendship', 'surrogate family']),
    "enemies-to-lovers": (['enemies to lovers'], ['rivalry']),
    "forbidden-love": (['forbidden love'], ['interracial romance']),
    "survival": (['survival'], ['stranded', 'shipwreck', 'castaway', 'survivalism']),
    "revenge": (['revenge', 'vengeance'], ['vendetta', 'retribution', 'payback']),
    "redemption": (['redemption'], ['second chance', 'atonement', 'forgiveness']),
    "man-vs-nature": (['man vs nature'], ['wilderness', 'natural disaster']),
    "political": (['politics', 'political'], ['election', 'government', 'propaganda', 'revolution']),
    "political-intrigue": (['political intrigue'], ['espionage', "coup d'etat", 'spy', 'assassination']),
    "class-struggle": (['class differences', 'social class'], ['poverty', 'working class', 'capitalism', 'social inequality']),
    "power": (['power'], ['corruption', 'greed', 'ambition', 'tyranny']),
    "legacy": (['legacy', 'inheritance'], ['family business', 'dynasty']),
    "identity": (['identity', 'identity crisis'], ['self-discovery', 'double life', 'dual identity']),
    "grief": (['grief', 'mourning'], ['loss of loved one', 'widow', 'widower', 'funeral', 'bereavement']),
    "family": (['family', 'family relationships'], ['dysfunctional family', 'family drama', 'siblings', 'parent child relationship']),
    "existential": (['existentialism'], ['meaning of life', 'nihilism', 'absurdism']),
    "psychological": (['psychological thriller', 'psychology'], ['mental illness', 'mental health', 'psychiatry']),
    "queer-rep": (['lgbt', 'gay theme', "boys' love (bl)"], ['gay', 'lesbian', 'transgender', 'homosexuality', 'coming out', 'queer']),

    # ---- movie-only craft ----
    "black-and-white": (['black and white'], ['silent film']),
    "documentary-style": (['mockumentary', 'documentary'], ['fake documentary', 'docudrama']),
    "ensemble-cast": ([], ['multiple storylines', 'anthology']),
    "found-footage": (['found footage'], ['fake documentary']),
    "long-take": (['one shot', 'long take'], ['single take']),
    "musical": (['musical'], ['song and dance', 'broadway', 'opera', 'concert']),
    "animation": (['anime', '3d animation'], ['stop motion', 'claymation', 'hand drawn animation', 'cartoon']),
    "practical-effects": ([], ['stop motion', 'special effects', 'claymation']),
    "stylized-violence": (['stylized violence'], ['gore', 'bloody', 'brutality']),
}

SCORE_PRIMARY = 0.7
SCORE_SUPPORTING = 0.5


def tags_for(keywords, allowed=None):
    """Map an iterable of raw TMDB keywords to [(tag, score), ...].

    keywords : list/ndarray of keyword strings, or a comma-joined string
    allowed  : optional set restricting which curated tags may be emitted
               (movies = ALL_MOVIE_TAGS, TV = ALL_TV_TAGS)

    NB: do NOT write `keywords or []`. keywords_tmdb arrives as a numpy array
    and `or` calls __bool__ on it -> ambiguous-truth-value ValueError.
    """
    if keywords is None:
        return []
    if isinstance(keywords, str):
        keywords = keywords.split(",")
    else:
        try:
            keywords = list(keywords)
        except TypeError:
            return []                      # NaN / scalar -> no keywords
    kws = {str(k).strip().lower() for k in keywords if str(k).strip()}
    if not kws:
        return []
    out = []
    for tag, (primary, supporting) in KEYWORD_MAP.items():
        if allowed is not None and tag not in allowed:
            continue
        if any(p in kws for p in primary):
            out.append((tag, SCORE_PRIMARY))
        elif any(s in kws for s in supporting):
            out.append((tag, SCORE_SUPPORTING))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out
