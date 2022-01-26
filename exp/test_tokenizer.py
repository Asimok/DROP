from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
a = 'How many points were Pittsburgh leading by at the start of the 2nd half?'
b = [
    'a',
    'How',
    'many',
    'points',
    'were',
    'Pittsburgh',
    'leading',
    'by',
    'at',
    'the',
    'start',
    'of',
    'the',
    '2nd',
    'half',
    '?',
    '[SEP]',
    '[CLS]'
]
c = 'and improved their all-time series record against the Browns to 22-5 (including the playoffs).'
d = "The Steelers entered their first matchup with the Cleveland Browns having won 15 of the last 16 games between the two teams. Ben Roethlisberger started the game despite missing practice time during the week due to a shoulder injury suffered in Week One. The Steelers' defense held the Browns to 16 yards on their first four drives, as the teams played to a scoreless tie after the first quarter. After a Bryant McFadden interception the Steelers drove 70 yards and Roethlisberger connected with Hines Ward for their third touchdown combination of the season. Cleveland responded with a 14 play, 71 yard drive, but Troy Polamalu intercepted a Cleveland pass as time expired in the first half solidifying Pittsburgh's seven point halftime lead. A 48-yard pass from Roethlisberger to Santonio Holmes and a 48-yard field goal from Jeff Reed on the team's second drive of the second half brought the score to 10-0. The Browns' Phil Dawson converted two consecutive field goals, to pull Cleveland within four points with 3:21 remaining. After a fourth down stop, Cleveland's offense took over with 26 seconds remaining, but failed to gain yardage as time expired. With the win, the Steelers increased their win streak over the Browns to 10 consecutive games—the longest current winning streak over a single opponent in the NFL. With the win the Steelers improved to 2-0 and led the AFC North by 1/2 a game ahead of the Ravens."
tokenizer.convert_tokens_to_ids(b)

# tokenizer.decode(
#     [101, 1998, 5301, 2037, 2035, 1011, 2051, 2186, 2501, 2114, 1996, 13240, 2000, 2570, 1011, 1019, 1006, 2164, 1996,
#      7555, 1007, 1012, 102])
# m = tokenizer('matchup')
# m.encodings[0].tokens

m = tokenizer(d)