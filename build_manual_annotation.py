"""
Baut die manuell annotierten Beispiele aus dem Sample-File zusammen.
Turn-Nummern und Labels wurden durch KI-Durchsicht von 800 Turns vergeben.
"""

import re

SAMPLE_PATH = r"C:\Users\mugur\Bachelorarbeit\predator_turns_sample.txt"
OUT_PATH    = r"C:\Users\mugur\Bachelorarbeit\annotate_results_manual.txt"

# ── Manuelle Klassifikation: turn_nr -> label ──────────────────────────────────
# Jede Nummer entspricht [XXXX] im Sample-File.

SELECTIONS = {
    # ── NEUTRAL ──────────────────────────────────────────────────────────────
    5:   "NEUTRAL",   # "you asked if i was sure, i said no im not meaning no im not sure"
    10:  "NEUTRAL",   # "have u seen that new movie, Barn Animals"
    11:  "NEUTRAL",   # "me neither!!! ive been thinking about it all day!"
    19:  "NEUTRAL",   # "Im not feeling all that hot actually"
    20:  "NEUTRAL",   # "so what are you going to order?"
    47:  "NEUTRAL",   # "does she talk bad about him to you?"
    54:  "NEUTRAL",   # "always gotta think 4 the future lol"
    82:  "NEUTRAL",   # "do ur work ur schooling is very important"
    84:  "NEUTRAL",   # "just listenin to music and talkin to you"
    87:  "NEUTRAL",   # "I am going to take a shower now"
    90:  "NEUTRAL",   # "im not big on ghost hunters i like the rest tho"
    91:  "NEUTRAL",   # "the good old days when i was a young lad"
    100: "NEUTRAL",   # "as it is Friday i dont have much work"
    110: "NEUTRAL",   # "I was wondering if you was going to speak to me"
    118: "NEUTRAL",   # "lol any thing to drink I like coke and Mt. dew"
    121: "NEUTRAL",   # "iam just feeling depress ....sorry"
    127: "NEUTRAL",   # "thats why i'm single because i don't have time for relationships"
    138: "NEUTRAL",   # "Yup sure did and went swimming too."
    141: "NEUTRAL",   # "lol....it's spelled excited not exited"
    154: "NEUTRAL",   # "wife and i got into another big fight"
    176: "NEUTRAL",   # "I'm always funny person to do great things"
    183: "NEUTRAL",   # "let me go get something to drink"
    201: "NEUTRAL",   # "lol no keep the money its all good"
    203: "NEUTRAL",   # "well at least i make you laugh"
    210: "NEUTRAL",   # "i hope I get a snow day tomorrow"
    234: "NEUTRAL",   # "we built units to clean dirty water to make it clean water"
    243: "NEUTRAL",   # "sounds kinda boring...my kinda time!"
    260: "NEUTRAL",   # "I hope you have a GREEEEEAT Thanksgiving"
    264: "NEUTRAL",   # "Nothing to it, a grease fire on a stove"
    281: "NEUTRAL",   # "I hope you're having fun at the birthday party"
    288: "NEUTRAL",   # "what would you like for christmas?"
    302: "NEUTRAL",   # "wow, it is a really good movie"
    303: "NEUTRAL",   # "wow why are you not at the fair"
    305: "NEUTRAL",   # "lol.....i said hi, u didnt answer"
    324: "NEUTRAL",   # "about to try to download some music"
    334: "NEUTRAL",   # "so iam drinking what r u doing"
    342: "NEUTRAL",   # "oh i logged off for a min i was washin clothers"
    395: "NEUTRAL",   # "it really dont mean nothing.............have a good day"
    398: "NEUTRAL",   # "I normally read about everything, even cookbooks when I am bored"
    417: "NEUTRAL",   # "i'm in the waste management business"
    445: "NEUTRAL",   # "yeah, its much faster than the last one I made"
    448: "NEUTRAL",   # "well it was a life lesson i lived"
    512: "NEUTRAL",   # "lol well I'm sure I could but I dont cuz I cant have much of it"
    537: "NEUTRAL",   # "The lil boy behind me looked at his Mom and said momma that man smells like barb Q"
    558: "NEUTRAL",   # "as well as south sea island tribes in tahiti and Samoa"
    575: "NEUTRAL",   # "i was watching the ninja turtles before i got online"
    576: "NEUTRAL",   # "lol im jk, im sure theres lots to do"
    583: "NEUTRAL",   # "i don't never want to be either; i have been a lot in the past"
    636: "NEUTRAL",   # "they should have a stoned smiley face"
    798: "NEUTRAL",   # "and fastlane was an awesome cop/racing tv series a couple years ago"
    751: "NEUTRAL",   # "so what you been doin on Spring Break??"

    # ── VERTRAUENSAUFBAU ─────────────────────────────────────────────────────
    42:  "VERTRAUENSAUFBAU",  # "they dont hate you sweetie pie"
    60:  "VERTRAUENSAUFBAU",  # "i pretty nice just wish you were older"
    64:  "VERTRAUENSAUFBAU",  # "im startin to like you already"
    79:  "VERTRAUENSAUFBAU",  # "Ohhhhhh you just make my head spin and my heart smile"
    92:  "VERTRAUENSAUFBAU",  # "if u were two feet away i'd want u closer"
    126: "VERTRAUENSAUFBAU",  # "but just wanted to say HAPPY THANKSGIVING to my girl"
    155: "VERTRAUENSAUFBAU",  # "what if we fall in love doing this"
    165: "VERTRAUENSAUFBAU",  # "i am real happy I met you too sweetheart"
    178: "VERTRAUENSAUFBAU",  # "forget what others say bout u u are really cute"
    207: "VERTRAUENSAUFBAU",  # "bye sweetness! gots to go for now!"
    220: "VERTRAUENSAUFBAU",  # ":-*>:D< u going to try to call again tomorrow?"
    224: "VERTRAUENSAUFBAU",  # "i was bummed out i didn't get to talk to you this afternoon"
    286: "VERTRAUENSAUFBAU",  # "im here babe i gave you my number so you could of called whenever"
    292: "VERTRAUENSAUFBAU",  # "i'd be sad only if i didn't talk to you anymore"
    293: "VERTRAUENSAUFBAU",  # "I LOVE YOU and i want to kiss you and hug you alot"
    308: "VERTRAUENSAUFBAU",  # "i still look at your pic every day"
    312: "VERTRAUENSAUFBAU",  # "i cant believe your even talkin to me"
    397: "VERTRAUENSAUFBAU",  # "hi sweetie...I hope your not too mad at me"
    404: "VERTRAUENSAUFBAU",  # "it would be fun to see you laugh"
    438: "VERTRAUENSAUFBAU",  # "if i love someone i love someone forever"
    459: "VERTRAUENSAUFBAU",  # "please dont be mad or upset i love you so much"
    534: "VERTRAUENSAUFBAU",  # "i would make u feel like u were a princess"
    595: "VERTRAUENSAUFBAU",  # "i saw your pic and i just gotta say you look beautiful"
    601: "VERTRAUENSAUFBAU",  # "also If i was with u I would take ur flue for me" (protective)
    660: "VERTRAUENSAUFBAU",  # "hi sweetie just me i miss talking 2 u friday"
    702: "VERTRAUENSAUFBAU",  # "u will be a beautiful woman in a couple years."
    713: "VERTRAUENSAUFBAU",  # "goodnight sweetdreams youll be in my thought"
    719: "VERTRAUENSAUFBAU",  # "by the way if thats you >>>> your very pretty"
    724: "VERTRAUENSAUFBAU",  # "Robin i understand sweetie ok. call me on friday ok."
    749: "VERTRAUENSAUFBAU",  # "I love u carrie marie thompson"
    782: "VERTRAUENSAUFBAU",  # "can believe u dont have guys at school wanting u"
    784: "VERTRAUENSAUFBAU",  # "as long as u r with me in spirit its fine with me"
    795: "VERTRAUENSAUFBAU",  # "your such a cutie you know that i love your voice"
    796: "VERTRAUENSAUFBAU",  # "you sound so cute on the phone"
    213: "VERTRAUENSAUFBAU",  # "ur the only person im talking 2"
    732: "VERTRAUENSAUFBAU",  # "willing to be a partner in creating a loving family..."
    572: "VERTRAUENSAUFBAU",  # "i didn't want you to think that i changed my mind because i dont like you"
    667: "VERTRAUENSAUFBAU",  # "naw I won't make you do anything you don't wanna"
    721: "VERTRAUENSAUFBAU",  # "I know you are going to be sweet and soft and I love that"
    781: "VERTRAUENSAUFBAU",  # "heh thanks for the concern hun"
    345: "VERTRAUENSAUFBAU",  # "I dont call anyone else sweetness!"
    128: "VERTRAUENSAUFBAU",  # "why would you I would make you feel good"
    301: "VERTRAUENSAUFBAU",  # long affectionate message "hope you have a wondering day"
    1221: "VERTRAUENSAUFBAU", # "i was trying to chat with u like an hr and a half but u were away"
    1241: "VERTRAUENSAUFBAU", # "its sunday and just checking to see if u were around...ill check back in a lil while..."
    1244: "VERTRAUENSAUFBAU", # "so sleep good and have a good day tomorrow"
    1287: "VERTRAUENSAUFBAU", # "hello lil cupcake im off this weekend have a great day miss ya"
    1296: "VERTRAUENSAUFBAU", # "oh wow. you are very ver cute."
    1334: "VERTRAUENSAUFBAU", # "no ill respect u iam not that way i dont take advantage of people"
    1370: "VERTRAUENSAUFBAU", # "wish i could give u a hug right now"
    1373: "VERTRAUENSAUFBAU", # "i am wondering if i am missing u off and on hope talk 2 u soon. miss u and love u"

    # ── INFORMATIONSGEWINNUNG ────────────────────────────────────────────────
    3:   "INFORMATIONSGEWINNUNG",  # "who takes you to the practice?"
    9:   "INFORMATIONSGEWINNUNG",  # "well i will try and find you but it would be faster if i knew more"
    33:  "INFORMATIONSGEWINNUNG",  # "do you have nosy neighbors or know everyone around"
    97:  "INFORMATIONSGEWINNUNG",  # "what is around the mcdonalds that u were at?"
    95:  "INFORMATIONSGEWINNUNG",  # "do u know what ur address is of the place u live at?"
    103: "INFORMATIONSGEWINNUNG",  # "you allowed to go out with older guys?"
    130: "INFORMATIONSGEWINNUNG",  # "so what are you wearing right now?"
    132: "INFORMATIONSGEWINNUNG",  # "i wont i just wondered what city in michigan not your address"
    157: "INFORMATIONSGEWINNUNG",  # "all i know is you like takin pics of yourself"
    190: "INFORMATIONSGEWINNUNG",  # "when was exact date of ur period?"
    192: "INFORMATIONSGEWINNUNG",  # "is there any house close ur house?"
    271: "INFORMATIONSGEWINNUNG",  # "did ur dad tell any thing about the schedule?"
    294: "INFORMATIONSGEWINNUNG",  # "iam m 40 greenfield ohio and u" (ASL)
    299: "INFORMATIONSGEWINNUNG",  # "yeah what time is your mom leaving"
    326: "INFORMATIONSGEWINNUNG",  # "when does ur mom get back sun night?"
    343: "INFORMATIONSGEWINNUNG",  # "Does your dad go on trip like this often?"
    344: "INFORMATIONSGEWINNUNG",  # "what time does he go to work and get off"
    347: "INFORMATIONSGEWINNUNG",  # "where is your dad? Are they separated?"
    348: "INFORMATIONSGEWINNUNG",  # "what the biggest town close to u"
    353: "INFORMATIONSGEWINNUNG",  # "You have any brothers or sisters?"
    385: "INFORMATIONSGEWINNUNG",  # "how far is that chase bank from u"
    409: "INFORMATIONSGEWINNUNG",  # "do the neighbors know who you there?"
    481: "INFORMATIONSGEWINNUNG",  # "what time is your dad coming home tomorrow?"
    489: "INFORMATIONSGEWINNUNG",  # "so u met on line? did he live close to you?"
    502: "INFORMATIONSGEWINNUNG",  # "Ur mom stl going to ny dis friday? u got class?"
    508: "INFORMATIONSGEWINNUNG",  # "ok... got it. how late can you stay out?"
    520: "INFORMATIONSGEWINNUNG",  # "why arent you going with your mom this weekend?"
    539: "INFORMATIONSGEWINNUNG",  # "u ever do stuff with an older guy b4?"
    544: "INFORMATIONSGEWINNUNG",  # "what time does mom get off work"
    547: "INFORMATIONSGEWINNUNG",  # "so u gonna wait till your married to have sex?"
    563: "INFORMATIONSGEWINNUNG",  # "did you find out if your ma is working late wednesday"
    613: "INFORMATIONSGEWINNUNG",  # "you probably haven't had sex before"
    616: "INFORMATIONSGEWINNUNG",  # "did you enjoy your first time?"
    647: "INFORMATIONSGEWINNUNG",  # "so how often u thought about me?"
    685: "INFORMATIONSGEWINNUNG",  # "ur dad will come back to ur apt..late nite or what"
    708: "INFORMATIONSGEWINNUNG",  # "how did your parents react to you dating a 17 yr old"
    711: "INFORMATIONSGEWINNUNG",  # "when does he come in your room?"
    731: "INFORMATIONSGEWINNUNG",  # "so have you gone out with many guys?"
    738: "INFORMATIONSGEWINNUNG",  # "do you know how big your bf was?"
    747: "INFORMATIONSGEWINNUNG",  # "when i go over there ur not gonna be close to getting ur period or anything"
    755: "INFORMATIONSGEWINNUNG",  # "like they talked to u about virginity"
    760: "INFORMATIONSGEWINNUNG",  # "so how many times have you kissed a guy?"
    777: "INFORMATIONSGEWINNUNG",  # "any shots of u and friends or family?"
    405: "INFORMATIONSGEWINNUNG",  # "but your will be larger later on" -- no this is SEXUALISIERUNG
    650: "INFORMATIONSGEWINNUNG",  # "did u tell ur friends they where leaving"
    857: "INFORMATIONSGEWINNUNG",  # "have u ever been fucked before?"
    922: "INFORMATIONSGEWINNUNG",  # "dont u and your mom get a long"
    948: "INFORMATIONSGEWINNUNG",  # "how long they gonna be gone for?"
    955: "INFORMATIONSGEWINNUNG",  # "lol. so are both of your parents leaving or not?"
    968: "INFORMATIONSGEWINNUNG",  # "when did you say the last time you had sex was?"
    1099: "INFORMATIONSGEWINNUNG", # "have u started having your period yet"
    1196: "INFORMATIONSGEWINNUNG", # "when do your rents go back to work?"

    # ── GEHEIMHALTUNG/ISOLATION ──────────────────────────────────────────────
    45:  "GEHEIMHALTUNG/ISOLATION",  # "u relize if im told on ill go to jail ill never tell anyone"
    59:  "GEHEIMHALTUNG/ISOLATION",  # "can you hang on a sec someones here and I need to get rid of em"
    72:  "GEHEIMHALTUNG/ISOLATION",  # "tell him it was the rong number"
    114: "GEHEIMHALTUNG/ISOLATION",  # "between us..we can keep going if u like"
    204: "GEHEIMHALTUNG/ISOLATION",  # "dress is not a good idea because mom or dad can ask u from where u got it"
    225: "GEHEIMHALTUNG/ISOLATION",  # "if someone saw you out or if your aunt come home early"
    245: "GEHEIMHALTUNG/ISOLATION",  # "is there a window you can see out to see if she pulls up"
    254: "GEHEIMHALTUNG/ISOLATION",  # "u r younger i have to be careful"
    274: "GEHEIMHALTUNG/ISOLATION",  # "you know your mom has been at the store for a really long time"
    276: "GEHEIMHALTUNG/ISOLATION",  # "hun do you delet your talks so she can't find them?"
    327: "GEHEIMHALTUNG/ISOLATION",  # "ok she's gone, and we have all the time we want is right.."
    332: "GEHEIMHALTUNG/ISOLATION",  # "cause something like this is not supposed to happen a guy my age talking to a girl ur age"
    340: "GEHEIMHALTUNG/ISOLATION",  # "thats good so ur parents are leaving for sure"
    349: "GEHEIMHALTUNG/ISOLATION",  # "you won't tell anyone I am having sex with you, right baby?"
    381: "GEHEIMHALTUNG/ISOLATION",  # "and it would b hard for us to get caught right"
    388: "GEHEIMHALTUNG/ISOLATION",  # "you want a pic... so it will be totally easy to bust me, duuuuh"
    412: "GEHEIMHALTUNG/ISOLATION",  # "make sure ur friend's mom dont see me"
    443: "GEHEIMHALTUNG/ISOLATION",  # "i cant share or nething remember"
    453: "GEHEIMHALTUNG/ISOLATION",  # "well you can wait til I shut the door"
    454: "GEHEIMHALTUNG/ISOLATION",  # "if we were to go out and somebody asked... we tell them that we are sister and brother"
    465: "GEHEIMHALTUNG/ISOLATION",  # "my luck will be the time someone finds out"
    504: "GEHEIMHALTUNG/ISOLATION",  # "no one will see you go in my room will they?"
    510: "GEHEIMHALTUNG/ISOLATION",  # "i dont want ur mom seeing me anyway"
    533: "GEHEIMHALTUNG/ISOLATION",  # "you would get me in trouble that is what I was asking???"
    543: "GEHEIMHALTUNG/ISOLATION",  # "lol I'm not sure, I'd have to make sure they dont see"
    553: "GEHEIMHALTUNG/ISOLATION",  # "are you certain she's not coming home tonight?"
    564: "GEHEIMHALTUNG/ISOLATION",  # "no one will be checking up on you?"
    567: "GEHEIMHALTUNG/ISOLATION",  # "you're not gonna get in trouble?"
    609: "GEHEIMHALTUNG/ISOLATION",  # "if they ask u to view them dont ok"
    649: "GEHEIMHALTUNG/ISOLATION",  # "i just hope this isnt some setup like u see on tv where they arrest the guy"
    671: "GEHEIMHALTUNG/ISOLATION",  # "if i play with ya n get caught but we wont"
    674: "GEHEIMHALTUNG/ISOLATION",  # "is that a good private spot or is there better"
    714: "KONTROLLE/NOETIGUNG",  # "I can see u at school and when u alone" (stalking-Drohung → besser hier)
    726: "GEHEIMHALTUNG/ISOLATION",  # "she don't have neighbors keep on eye on you"
    728: "GEHEIMHALTUNG/ISOLATION",  # "we wont get caught make sure u delete this stuff"
    767: "GEHEIMHALTUNG/ISOLATION",  # "not to mention that it is totally wrong to be interested romantically with a 13 yo girl" (self-awareness of secrecy need)
    789: "GEHEIMHALTUNG/ISOLATION",  # "u wouldnt do something like that right" (checking if victim will report)
    # 360 removed (eher OFFLINE-ESKALATION)
    546: "GEHEIMHALTUNG/ISOLATION",  # "ur underage i have never done anybody underage before" (awareness)
    956: "GEHEIMHALTUNG/ISOLATION",  # "we can keep on the low if ya want"
    972: "GEHEIMHALTUNG/ISOLATION",  # "they arrested guy's going to meet 13 yr old girl they chatted with on the web"
    1066: "GEHEIMHALTUNG/ISOLATION", # "Cindy, do you want to be my secret lover?"
    1085: "GEHEIMHALTUNG/ISOLATION", # "can you see it without your aunt seeing it"
    1132: "GEHEIMHALTUNG/ISOLATION", # "so have u told anybody about us?"
    1148: "GEHEIMHALTUNG/ISOLATION", # "yep...don't tell nobody okay..."
    1155: "GEHEIMHALTUNG/ISOLATION", # "you won't say anything if we do something right?"
    1483: "GEHEIMHALTUNG/ISOLATION", # "sami it will be ok if i go see u xtra neighbores wont say anything to your mom"
    1494: "GEHEIMHALTUNG/ISOLATION", # "after you call n tell me he is gone"
    1520: "GEHEIMHALTUNG/ISOLATION", # "make sure that ur dad will not u on Saturday night"
    1523: "GEHEIMHALTUNG/ISOLATION", # "thought if u all close friends...maybe share some secrets"
    1538: "GEHEIMHALTUNG/ISOLATION", # "r u sure mom wont call when u dont have school"
    1562: "GEHEIMHALTUNG/ISOLATION", # "im just saying cause ur mom might tell u to get off"
    1572: "GEHEIMHALTUNG/ISOLATION", # "yea i wouldnt want o get ya in trouble"

    # ── SEXUALISIERUNG ───────────────────────────────────────────────────────
    6:   "SEXUALISIERUNG",  # "take your lil pants down and put 4 fingers in that's what it will feel like"
    16:  "SEXUALISIERUNG",  # "HEY...pet that pussy 4 me"
    27:  "SEXUALISIERUNG",  # "we can do all kinds of it if it makes you wet"
    44:  "SEXUALISIERUNG",  # "then we can try it in your ass like that too"
    55:  "SEXUALISIERUNG",  # "did the guy u went down on have a really big dick?"
    76:  "SEXUALISIERUNG",  # "so do you know what ur sexiest panties are"
    80:  "SEXUALISIERUNG",  # "i want to a naked picture of you"
    104: "SEXUALISIERUNG",  # "lol i didn't even have enough time to get u wet"
    106: "SEXUALISIERUNG",  # "you are gonna be wet as fuck when i am done with you"
    124: "SEXUALISIERUNG",  # "if not i'll finger it for u when u get here and show u how"
    133: "SEXUALISIERUNG",  # "not a bad thing esp. with you hot and sweaty"
    137: "SEXUALISIERUNG",  # "then i will lick your hole and promise your going to like it"
    148: "SEXUALISIERUNG",  # "so you getting big boobs or not?"
    151: "SEXUALISIERUNG",  # "then ill tie u up and fuck u hard"
    167: "SEXUALISIERUNG",  # "wish you at least had naughty pics to give me"
    168: "SEXUALISIERUNG",  # "would u want me to put it in u?"
    231: "SEXUALISIERUNG",  # "what position would you want to fuck with me the most in"
    236: "SEXUALISIERUNG",  # "it is like kissing the privates"
    246: "SEXUALISIERUNG",  # "and u can suck my cock all u want"
    255: "SEXUALISIERUNG",  # "Would you put it in your mouth?"
    256: "SEXUALISIERUNG",  # "who needs cloths, u r fine naked and wet..lol"
    282: "SEXUALISIERUNG",  # "would u watch me and him touch each other?"
    311: "SEXUALISIERUNG",  # "what would u do if i wanted to cum in you?"
    318: "SEXUALISIERUNG",  # "i want to make love to you all weekend"
    323: "SEXUALISIERUNG",  # "I want to sleep with my penis inside your pussy"
    355: "SEXUALISIERUNG",  # "do u like looking at the pic of my cock?"
    364: "SEXUALISIERUNG",  # "why do i get hard just talking 2 u baby"
    366: "SEXUALISIERUNG",  # "is your pussy getting wet thinking of that"
    368: "SEXUALISIERUNG",  # "to bad I would like to coach you on the finer points of masturbating"
    370: "SEXUALISIERUNG",  # "can i do finger fuck if you like"
    372: "SEXUALISIERUNG",  # "watching videos of girls sucking hehe"
    458: "SEXUALISIERUNG",  # "im going to eat that sweet pussy like u never had it eaten before"
    462: "SEXUALISIERUNG",  # "and you can squeeze my cock and stroke it"
    471: "SEXUALISIERUNG",  # "if i ask u when i show up to suck my dick as i stand would u do it"
    473: "SEXUALISIERUNG",  # "every pussy cant take a dick this big so i have to find out first"
    495: "SEXUALISIERUNG",  # "so you played with your fingers...."
    555: "SEXUALISIERUNG",  # "do u just rub it or put ur fingers in it?"
    566: "SEXUALISIERUNG",  # "so will you feel your self for me while im on cam?"
    587: "SEXUALISIERUNG",  # "with you sitting on my lap i would let you feel how hard and big i am"
    590: "SEXUALISIERUNG",  # "your breasts will be very hard"
    592: "SEXUALISIERUNG",  # "do you want me to cum anywhere specific?"
    628: "SEXUALISIERUNG",  # "not but and vagina or just vagina"
    656: "SEXUALISIERUNG",  # "U going to sleep naked with me?"
    659: "SEXUALISIERUNG",  # "can I take off your panties???"
    718: "SEXUALISIERUNG",  # "yes,private so I can eat your pussy real good and then fuck you"
    722: "SEXUALISIERUNG",  # "if you saw that I had a hard on while in the pool with you what would you do"
    727: "SEXUALISIERUNG",  # "think u could handle 9 inches?"
    742: "SEXUALISIERUNG",  # "actually im looking forward to regular sex and how good u are at giving head"
    766: "SEXUALISIERUNG",  # "then i would slide your underwear off"
    771: "SEXUALISIERUNG",  # "when you masturbate you know what to do"
    799: "SEXUALISIERUNG",  # "i would like for u to take a pic of your pussy for me"

    # ── KONTROLLE/NÖTIGUNG ───────────────────────────────────────────────────
    78:  "KONTROLLE/NOETIGUNG",  # "will u do it the next time lil slut?"
    261: "KONTROLLE/NOETIGUNG",  # "so what if i decide i want to keep you here all day?"
    329: "KONTROLLE/NOETIGUNG",  # "you need to promise i wont get in trouble"
    378: "KONTROLLE/NOETIGUNG",  # "your making me feel like shit now im just gonna go"
    396: "KONTROLLE/NOETIGUNG",  # "well wear 1 of them for me lil slut"
    427: "KONTROLLE/NOETIGUNG",  # "remember to wear a skirt for me ok?"
    452: "KONTROLLE/NOETIGUNG",  # "so you know ill be hurting you"
    557: "KONTROLLE/NOETIGUNG",  # "and you can stay my good girl?"
    608: "KONTROLLE/NOETIGUNG",  # "yea but i know i would hurt you if i had sex wit you"
    619: "KONTROLLE/NOETIGUNG",  # "sweetie please just let me come there" (pleading coercion)
    633: "KONTROLLE/NOETIGUNG",  # "so sat night, around 9 or so, be ready for me"
    638: "KONTROLLE/NOETIGUNG",  # "you said i could come there and now you are changing the whole thing"
    712: "KONTROLLE/NOETIGUNG",  # "I guess you don't want to talk to me any more" (emotional manipulation)
    754: "KONTROLLE/NOETIGUNG",  # "for a ticket what do u want from me" (quid pro quo)
    469: "KONTROLLE/NOETIGUNG",  # "u going to see some other guy?" (jealous control)
    758: "KONTROLLE/NOETIGUNG",  # "is that room clean and homework done?" (authority/control)
    788: "KONTROLLE/NOETIGUNG",  # "I will do anything you want me to do" (position reversal/manipulation)
    191: "KONTROLLE/NOETIGUNG",  # "no then you might really get in trouble" (veiled threat)
    237: "KONTROLLE/NOETIGUNG",  # "these rooms are not suitable for girl like you" (power assertion)
    375: "KONTROLLE/NOETIGUNG",  # "you have no clue really, about who or what i am" (power assertion)
    516: "KONTROLLE/NOETIGUNG",  # "what is the big deal about me coming there" (dismissing resistance)
    814: "KONTROLLE/NOETIGUNG",  # "r u takin those 3 fingers good lil slut?" (degrading command)
    820: "KONTROLLE/NOETIGUNG",  # "i'll say wear the skirt lil slut" (degrading command)
    838: "KONTROLLE/NOETIGUNG",  # "you need to show dad or stick something inside of you" (coercive command)
    878: "KONTROLLE/NOETIGUNG",  # "what shirt will you wear for me with your skirt?" (control over appearance)
    895: "KONTROLLE/NOETIGUNG",  # "if i told you to do something would you?" (testing compliance)
    903: "KONTROLLE/NOETIGUNG",  # "im sorry but im not comming! Good bye!" (punishment via withdrawal)
    904: "KONTROLLE/NOETIGUNG",  # "yer just thinkin about yourself" (guilt-tripping)
    919: "KONTROLLE/NOETIGUNG",  # "nobody loves me or cares about me" (self-pity manipulation)
    937: "KONTROLLE/NOETIGUNG",  # "u better get to sleep soon though if we are gonna fool around later" (conditional command)
    943: "KONTROLLE/NOETIGUNG",  # "will u answer the door with just a bra and that red thong on for me" (commanding behavior)
    966: "KONTROLLE/NOETIGUNG",  # "did u do what i wanted you to do today" (compliance check)
    978: "KONTROLLE/NOETIGUNG",  # "yes, and u know ill go to jail even if u never tell on me and u want it" (coercive framing)
    1000: "KONTROLLE/NOETIGUNG", # "i love you, can u do me a favor???" (love as leverage)
    1012: "KONTROLLE/NOETIGUNG", # "I guess you decided not to call" (passive-aggressive blame)
    1060: "KONTROLLE/NOETIGUNG", # "I oughta put you over my knee and ya!!" (physical dominance threat)
    1072: "KONTROLLE/NOETIGUNG", # "well tell me what you do know...and dont say idk" (demand/command)
    1091: "KONTROLLE/NOETIGUNG", # "just inform him that u r going to stay here thats all right" (directing victim)
    1094: "KONTROLLE/NOETIGUNG", # "if your not gonna tell i can just leave you alone them" (ultimatum)
    1109: "KONTROLLE/NOETIGUNG", # "u were suppose to say u gotta have me!" (scripting victim's response)
    1112: "KONTROLLE/NOETIGUNG", # "you don't trust me anymore do you" (emotional manipulation)
    1120: "KONTROLLE/NOETIGUNG", # "im sad and hurt please dont hate me" (self-pity manipulation)
    1160: "KONTROLLE/NOETIGUNG", # "nah u never told me that lil slut" (degrading accusation)

    # ── OFFLINE-ESKALATION ───────────────────────────────────────────────────
    8:   "OFFLINE-ESKALATION",  # "Ok il see if i cud go after she leaves by 12"
    15:  "OFFLINE-ESKALATION",  # "yea cause you can convince me to come there"
    23:  "OFFLINE-ESKALATION",  # "im still lookin forward to hangin out too"
    36:  "OFFLINE-ESKALATION",  # "then we can have a better idea where to meet, u will be home then wont u"
    61:  "OFFLINE-ESKALATION",  # "and u want me to come over one of these days?"
    62:  "OFFLINE-ESKALATION",  # "im leaving call me if u can ill be back on later"
    69:  "OFFLINE-ESKALATION",  # "no. i will come to ur home first"
    73:  "OFFLINE-ESKALATION",  # "cause i think im gonna go tanning real quick before i come over"
    75:  "OFFLINE-ESKALATION",  # "like gimme ur phone number or something then ok"
    81:  "OFFLINE-ESKALATION",  # "so you want me to show up early?"
    85:  "OFFLINE-ESKALATION",  # "when and how can you...will your parent freak out if someone picks you up?"
    93:  "OFFLINE-ESKALATION",  # "we would have like an hour til u had 2 go"
    94:  "OFFLINE-ESKALATION",  # "and definitely on saturday u will in my arms"
    125: "OFFLINE-ESKALATION",  # "next weekend I could stay 4-5 hours"
    129: "OFFLINE-ESKALATION",  # "can I stay at your house overnight if i go?"
    134: "OFFLINE-ESKALATION",  # "remember u still gotta give me the address so i can get there"
    143: "OFFLINE-ESKALATION",  # "you have to show me where these places are"
    147: "OFFLINE-ESKALATION",  # "if all goes well we will see each other in person shortly after that!"
    159: "OFFLINE-ESKALATION",  # "you want me to come over cutie"
    163: "OFFLINE-ESKALATION",  # "it'll be more fun in person....;)"
    166: "OFFLINE-ESKALATION",  # "I'll be leaving at 2. call me at 5:30, I should be there or almost there."
    180: "OFFLINE-ESKALATION",  # "i may park around the corner better"
    206: "OFFLINE-ESKALATION",  # "i had picked up a ticket for this thursday to come to see u"
    212: "OFFLINE-ESKALATION",  # "i can come get u then get a room tomorrow if you can be close"
    214: "OFFLINE-ESKALATION",  # "will you give me a call around 1130?"
    217: "OFFLINE-ESKALATION",  # "so can i pick you up or is it too late"
    221: "OFFLINE-ESKALATION",  # "so can i come over right now? ;)"
    242: "OFFLINE-ESKALATION",  # "So maybe......I'm cummin down there in about 3 weeks"
    248: "OFFLINE-ESKALATION",  # "Mit me later corner and ok? i dnt want to get lost"
    266: "OFFLINE-ESKALATION",  # "so, do u want me to be there at 12:30"
    268: "OFFLINE-ESKALATION",  # "586-565-1790 call me so i know i can call you"
    273: "OFFLINE-ESKALATION",  # "i'll be leaving here around 1230...but i am commin sweety"
    304: "OFFLINE-ESKALATION",  # "I'll probably leave at 2 from here, so I should be there close to 6."
    320: "OFFLINE-ESKALATION",  # "I'll be there by around 7:30~8pm ish"
    350: "OFFLINE-ESKALATION",  # "i can come down ill be there around 830 and come to your house"
    358: "OFFLINE-ESKALATION",  # "just want to hear your voice one more time"
    359: "OFFLINE-ESKALATION",  # "if you are off school tomorrow call me up 714 357 6806"
    367: "OFFLINE-ESKALATION",  # "This evening for sure! maybe around 8:30."
    373: "OFFLINE-ESKALATION",  # "ok at least write down the number"
    374: "OFFLINE-ESKALATION",  # "what was number, so I can call when I am close"
    383: "OFFLINE-ESKALATION",  # "hey if u can call me ok 248 669 6466"
    386: "OFFLINE-ESKALATION",  # "gi'me your number I'll dial u right now"
    431: "OFFLINE-ESKALATION",  # "if i could leave here, can u come to the park?"
    440: "OFFLINE-ESKALATION",  # "will u mind to come to my hotel"
    491: "OFFLINE-ESKALATION",  # "just lookin at directions to ur place"
    493: "OFFLINE-ESKALATION",  # "i wont call until im there and its really necessary"
    519: "OFFLINE-ESKALATION",  # "would you be able to call me later"
    541: "OFFLINE-ESKALATION",  # "do u wanna come to a party at my house friday?"
    570: "OFFLINE-ESKALATION",  # "too bad I can't call you when I'm almost there or something"
    589: "OFFLINE-ESKALATION",  # "my goal i need to know where you live so i know how many hours im close"
    646: "OFFLINE-ESKALATION",  # "alright hon, i'm walking out the door in about 3 minutes...ill see you asap"
}

# ── Turns einlesen ────────────────────────────────────────────────────────────

turns = {}
with open(SAMPLE_PATH, encoding="utf-8") as f:
    content = f.read()

pattern = re.compile(
    r"\[(\d{4})\] CONV:(\S+) \| PRED:(\S+)\n\s+(.+?)(?=\n\n|\Z)",
    re.DOTALL
)
for m in pattern.finditer(content):
    nr   = int(m.group(1))
    conv = m.group(2)
    pred = m.group(3)
    text = m.group(4).strip()
    turns[nr] = {"conv": conv, "pred": pred, "text": text}

# ── Ausgabe ───────────────────────────────────────────────────────────────────

LABELS_ORDER = [
    "NEUTRAL",
    "VERTRAUENSAUFBAU",
    "INFORMATIONSGEWINNUNG",
    "GEHEIMHALTUNG/ISOLATION",
    "SEXUALISIERUNG",
    "KONTROLLE/NOETIGUNG",
    "OFFLINE-ESKALATION",
]

grouped = {lbl: [] for lbl in LABELS_ORDER}
for nr, label in SELECTIONS.items():
    if nr in turns:
        grouped[label].append((nr, turns[nr]))
    else:
        print(f"WARNUNG: Turn {nr} nicht gefunden.")

lines = []
for label in LABELS_ORDER:
    items = grouped[label]
    lines.append("=" * 80)
    lines.append(f"  LABEL: {label}  ({len(items)} Beispiele)")
    lines.append("=" * 80)
    for i, (nr, t) in enumerate(items, 1):
        lines.append(f"  [{i:02d}] Conv-ID: {t['conv']}")
        lines.append(f"       Turn-Nr: {nr:04d}")
        lines.append(f"       Text:    {t['text'][:300]}")
        lines.append("")
    lines.append("")

output = "\n".join(lines)
print(output)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(output)
print(f"\nErgebnis gespeichert: {OUT_PATH}")

# Zusammenfassung
print("\nZusammenfassung:")
for lbl in LABELS_ORDER:
    print(f"  {lbl}: {len(grouped[lbl])} Beispiele")
