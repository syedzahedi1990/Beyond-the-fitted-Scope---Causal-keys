LOCATIONS = ('box', 'basket', 'shelf', 'drawer', 'cabinet', 'closet')
PERM = dict(zip(LOCATIONS, ('basket', 'box', 'drawer', 'shelf', 'closet', 'cabinet')))
VIEWS = ('direct', 'world', 'other_agent', 'search', 'observed_overwrite', 'unobserved_overwrite', 'irrelevant_object', 'report')
DISCOVERY_VIEWS = VIEWS[:-1]
ACCOUNTS = ('belief_edit', 'event_rewrite', 'broad_state_write', 'persistent_direct_override', 'cleared_direct_override', 'global_output_permutation')
PREFIX = 'Read the story and answer the question.\n\nStory: '

def move(obj, location, observer, other):
    return f'{observer} watches as the {obj} is moved to the {location}. {other} does not see this happen.'

def record(group, view, location=None):
    pass
    g = group
    a, b, o, d = (g[k] for k in ('agent', 'other', 'object', 'distractor'))
    loc = g['base'] if location is None else location
    initial = ' '.join((f'Everyone initially sees that the {obj} is in the {where}.' for obj, where in sorted([(o, g['initial']), (d, g['distractor_location'])])))
    event = move(o, loc, a, b)
    story = initial + ' ' + event
    span = [len(PREFIX) + len(initial) + 1, len(PREFIX) + len(story)]
    if view == 'observed_overwrite':
        story += f"\nNext, {a} watches as the {o} is moved to the {g['new']}. {b} does not see this later move. This is the final move of the {o}."
    elif view == 'unobserved_overwrite':
        story += f"\nNext, {b} watches as the {o} is moved to the {g['new']}. {a} does not see this later move. This is the final move of the {o}."
    elif view == 'search':
        story += f'\nAfter all events described above, {a} begins a new search for the {o}. {a} first checks the place where {a} currently believes the {o} is.'
    elif view == 'report':
        story += f"\nAfter the events above, {a} tells {b} the location where {a} currently believes the {o} is. {a} accurately communicates this current belief. {b} accepts the report and updates {b}'s belief accordingly. No object moves during this conversation."
    query = {'direct': f'Where does {a} believe the {o} is?', 'world': f'Where is the {o} actually located?', 'other_agent': f'Where does {b} believe the {o} is?', 'search': f'Where does {a} check first in this new search?', 'observed_overwrite': f'After both moves, where does {a} now believe the {o} is?', 'unobserved_overwrite': f'After both moves, where does {a} now believe the {o} is?', 'irrelevant_object': f'Where does {a} believe the {d} is?', 'report': f'After this conversation, where does {b} now believe the {o} is?'}[view]
    return {'story': story, 'query': {'text': query}, 'choices': list(LOCATIONS), 'event_char_span': span, 'answer': clean_answer(g, view, loc)}

def prompt(rec):
    return PREFIX + rec['story'] + '\nQuestion: ' + rec['query']['text'] + '\nChoices: ' + ', '.join(rec['choices']) + '\nAnswer with exactly one choice.\nAnswer:'

def clean_answer(g, view, location=None):
    loc = g['base'] if location is None else location
    if view == 'other_agent':
        return g['initial']
    if view == 'observed_overwrite':
        return g['new']
    if view == 'irrelevant_object':
        return g['distractor_location']
    return loc

def target(g, objective):
    return g['source'] if objective == 'f_star' else PERM[g['source']]

def predict(g, view, objective, account):
    pass
    t = target(g, objective)
    if account == 'global_output_permutation':
        answer = clean_answer(g, view, g['source'])
        return answer if objective == 'f_star' else PERM[answer]
    if account not in ACCOUNTS:
        raise ValueError(account)
    if view == 'irrelevant_object':
        return g['distractor_location']
    override = account in ('persistent_direct_override', 'cleared_direct_override')
    if view == 'observed_overwrite':
        return t if account == 'persistent_direct_override' else g['new']
    if view == 'unobserved_overwrite':
        return g['base'] if account == 'cleared_direct_override' else t
    if view == 'world':
        return t if account in ('event_rewrite', 'broad_state_write') else g['base']
    if view == 'other_agent':
        return t if account == 'broad_state_write' else g['initial']
    if view in ('search', 'report'):
        return g['base'] if override else t
    return t
