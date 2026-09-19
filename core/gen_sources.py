"""Deterministic pseudo-random generators that produce VARIED English
documents from large slot pools and many distinct template families.

Design rules:
  * one family = one leakage-control group (all its instances stay in the
    same train/val/test split);
  * every template is designed so any slot substitution stays grammatical;
  * families are capped so no single pattern dominates;
  * no code, no formatting gimmicks, just plain natural English;
  * RNG is seeded: the same corpus is reproducible from this file + seeds.

The generators produce STRUCTURAL variety — the hand-written seeds in
data/seeds/ provide the stylistic depth. Both are required.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List

# ------------------------------------------------------------------ pools --

FIRST_NAMES = [
    "Maya", "James", "Lena", "Mark", "Tom", "Anna", "Ben", "Ava", "Noah", "Grace",
    "Omar", "Priya", "Sam", "Clara", "David", "Emma", "Liam", "Sofia", "Ethan",
    "Isla", "Lucas", "Mia", "Henry", "Ruby", "Jake", "Nina", "Leo", "Zoe", "Max",
    "Ella", "Adam", "Lily", "Owen", "Ivy", "Felix", "Rose", "Hugo", "Alice",
    "Daniel", "Maria", "Victor", "Helen", "Peter", "Julia", "Marco", "Elena",
    "Simon", "Nora", "Paul", "Tessa", "Ravi", "Dana", "Ken", "Aisha", "Oliver",
    "Bella", "Chris", "Dora", "Elias", "Farah", "George", "Hana", "Iris", "Jonah",
    "Kate", "Luis", "Mona", "Nadiya", "Oscar", "Paula", "Ruth", "Stefan", "Tara",
    "Uma", "Vera", "Wendy", "Xavier", "Yara", "Zane", "Rosa", "Carl", "June",
]

PLACES = [
    "the park", "the library", "the market", "the beach", "the old bridge",
    "the train station", "the bakery on Elm Street", "the community garden",
    "the museum", "the swimming pool", "the lakeside path", "the town square",
    "the coffee shop", "the school gym", "the hardware store", "the riverbank",
    "the hill behind the church", "the bus stop near the fountain", "the harbor",
    "the bookstore", "the post office", "the farm stand", "the cinema",
    "the soccer field", "the mountain trail", "the harbor pier",
]

ACTIVITIES = [
    "a walk", "a picnic", "a game of chess", "some shopping", "a swim",
    "a bike ride", "a photography walk", "some gardening", "a kite flying session",
    "a cooking afternoon", "a puzzle night", "a fishing trip", "a painting session",
    "a reading hour", "a small hike", "a baking project", "a board game afternoon",
    "a bird watching walk", "a cleanup of the storage room", "a soccer match",
]

TOPICS_BOOKS = [
    "mystery novels", "travel books", "cookbooks", "history books", "poetry",
    "science magazines", "comic books", "biographies", "gardening guides",
    "adventure stories", "books about animals", "books about the sea",
    "old maps", "books about space", "fairy tales", "books about inventions",
]

WEATHER = [
    "sunny and warm", "cold and windy", "gray and drizzly", "hot and humid",
    "crisp and clear", "cloudy with a chance of rain", "misty in the morning",
    "bright but cold", "warm with a light breeze", "stormy",
]

FOODS = [
    "tomato soup", "fresh bread", "apple pie", "a cheese sandwich", "pancakes",
    "vegetable stew", "rice and beans", "a fruit salad", "baked potatoes",
    "chicken soup", "an omelet", "pasta with tomatoes", "a banana smoothie",
    "roasted carrots", "oatmeal with honey", "a green salad", "grilled cheese",
]

ANIMALS = [
    "dog", "cat", "rabbit", "horse", "sparrow", "squirrel", "duck", "goat",
    "chicken", "sheep", "fox", "owl", "frog", "turtle", "goldfish", "hamster",
    "donkey", "pigeon", "hedgehog", "butterfly", "bee", "ladybug", "deer",
    "crow", "seal", "penguin", "camel", "lizard", "snail", "ant",
]

OBJECTS = [
    "wooden chair", "brass key", "leather notebook", "glass jar", "wool blanket",
    "metal spoon", "clay pot", "paper map", "silver watch", "basket",
    "kerosene lamp", "enamel mug", "canvas bag", "copper kettle", "pocket knife",
    "leather belt", "straw hat", "iron gate", "porcelain cup", "cedar box",
]

ROOM_ITEMS = [
    "a round table by the window", "a shelf full of old books",
    "a fireplace with a blackened hearth", "a worn green sofa",
    "a clock that ticks loudly", "a vase of dried flowers",
    "a rug with a faded pattern", "two armchairs facing each other",
    "a desk covered in papers", "a lamp with a yellow shade",
    "a painting of a harbor", "baskets stacked in the corner",
]

PROFESSIONS = [
    "teacher", "baker", "doctor", "farmer", "carpenter", "nurse", "pilot",
    "librarian", "electrician", "cook", "fisherman", "gardener", "mechanic",
    "tailor", "painter", "bus driver", "shopkeeper", "vet", "musician",
    "firefighter", "plumber", "writer", "photographer", "scientist",
]

FEELINGS = [
    "tired but happy", "a little nervous", "excited about the weekend",
    "calm and content", "worried about the exam", "relieved it is over",
    "curious about the news", "proud of the work", "grateful for the help",
    "annoyed by the delay", "hopeful about tomorrow", "homesick",
]

# explanation source material (question templates, subject pools, answer cores)
WHY_SUBJECTS = [
    ("the sky is blue",
     "sunlight is made of many colors, and air scatters the blue part of the light in every direction, so the whole sky looks blue to us"),
    ("we see lightning before thunder",
     "light travels much faster than sound, so the flash reaches our eyes almost at once while the thunder takes seconds to arrive"),
    ("ice floats on water",
     "ice takes up more space than the same amount of liquid water, which makes it lighter for its size, so it stays on top"),
    ("leaves change color in autumn",
     "trees stop making the green color in their leaves as the days get shorter, and the yellow and red colors that were hidden all along finally show"),
    ("the moon changes shape",
     "we only see the part of the moon that the sun lights up, and as the moon moves around the Earth that bright part grows and shrinks"),
    ("metal feels colder than wood",
     "metal carries heat away from your hand quickly, while wood lets the heat stay, so metal feels colder even when both are the same temperature"),
    ("bread rises in the oven",
     "tiny living yeast cells make bubbles of gas inside the dough, and the heat makes those bubbles grow and then harden in place"),
    ("the sea is salty",
     "rain slowly washes salt out of rocks and soil, rivers carry it to the sea, and the salt stays behind when the water evaporates"),
    ("we get goosebumps",
     "tiny muscles at the base of each hair tighten when we are cold or scared, a reflex left over from when thicker hair helped our ancestors"),
    ("onions make us cry",
     "cutting an onion releases a gas that stings the eyes, and the eyes make tears to wash the sting away"),
    ("the heart beats faster during exercise",
     "the muscles need more oxygen when they work hard, so the heart pumps faster to deliver it through the blood"),
    ("echoes happen in large empty rooms",
     "sound bounces off hard walls like a ball, and when the walls are far away we hear the bounce as a separate sound"),
    ("a pot of water boils at high heat",
     "heat gives the water particles enough energy to escape as vapor, and the bubbles we see are pockets of that vapor rising"),
    ("sleep is important",
     "sleep gives the brain time to sort memories and repair the body, which is why a tired mind learns slowly"),
    ("day turns into night",
     "the Earth spins like a top, and night is simply the time when our part of the Earth faces away from the sun"),
    ("desert days are hot but nights are cold",
     "dry sand heats up fast in the sun and loses that heat just as fast after sunset, because there is little water to hold the warmth"),
]

HOW_SUBJECTS = [
    ("a refrigerator keeps food cold",
     "it pumps a special fluid through pipes; the fluid absorbs heat inside the box and releases it outside, so the inside stays cold"),
    ("a bicycle stays upright",
     "the spinning wheels and the rider's balance work together; small corrections of the handlebars keep the bike from tipping over"),
    ("a thermometer measures temperature",
     "the liquid inside expands when it is warm and shrinks when it is cold, and the scale turns that change into a number"),
    ("a battery powers a lamp",
     "a chemical reaction inside the battery pushes tiny charges through the wire, and the moving charges make the lamp glow"),
    ("a plane stays in the air",
     "the wings are shaped so the air pushes them upward as the plane moves forward, and the engines keep it moving fast enough"),
    ("a dam makes electricity",
     "falling water spins large wheels called turbines, and the spinning motion is turned into electric current"),
    ("seeds grow into plants",
     "a seed holds a tiny plant and a lunch box of food; with water and warmth the plant wakes up, pushes out a root, and reaches for the light"),
    ("ants find food",
     "a scout ant leaves a trail of scent on the way back to the nest, and the other ants follow the smell to the food"),
    ("a magnet picks up pins",
     "an invisible field around the magnet pulls on the metal and drags small steel objects toward it"),
    ("a clock keeps time",
     "something inside the clock swings or vibrates at a steady rate, and the gears count those beats and move the hands"),
    ("rain forms",
     "water rises into the air as vapor, cools high above the ground, gathers into drops, and falls when the drops are heavy enough"),
    ("a wound heals",
     "the blood first forms a protective crust, and then the skin slowly rebuilds itself underneath until the crust falls away"),
]

DIFF_PAIRS = [
    ("a lake", "a river",
     "a lake is still and surrounded by land, while a river is water that flows in one direction toward the sea"),
    ("salt", "sugar",
     "salt is a mineral that the body needs in small amounts, while sugar comes from plants and is used mainly for energy and sweetness"),
    ("a frog", "a toad",
     "a frog usually has smooth wet skin and long jumping legs, while a toad has drier bumpy skin and prefers walking"),
    ("a seed", "a nut",
     "a nut is a kind of seed wrapped in a hard shell, but not every seed has such a hard protective case"),
    ("walking", "running",
     "in walking one foot is always on the ground, but in running there are short moments when both feet are in the air"),
    ("weather", "climate",
     "weather is what the sky is doing today, while climate is the pattern of weather a place has over many years"),
    ("a boat", "a ship",
     "a boat is small enough to be carried or rowed easily, while a ship is a large vessel built for long journeys"),
    ("a hill", "a mountain",
     "a mountain rises much higher and has steeper, rockier sides, while a hill is gentler and usually covered with grass or trees"),
    ("a poem", "a story",
     "a poem uses rhythm and carefully chosen sounds in a short space, while a story tells about people and events over a longer time"),
    ("a fact", "an opinion",
     "a fact can be checked and shown to be true, while an opinion tells what someone thinks or feels about something"),
    ("an insect", "a spider",
     "an insect has six legs and three body parts, while a spider has eight legs and two body parts"),
    ("a journal", "a novel",
     "a journal records real events from someone's life, while a novel is an invented story"),
]

WHAT_IF = [
    ("it never rained again",
     "rivers and lakes would slowly shrink, crops would fail, and people and animals would have to move or find new ways to get water"),
    ("the sun did not rise tomorrow",
     "the world would grow dark and cold very quickly, plants would stop making food, and life would become impossible before long"),
    ("a person stopped drinking water",
     "the body would dry out its cells step by step, and after a few days the person would become seriously ill"),
    ("all the bees disappeared",
     "many plants would not be pollinated, so fruits and seeds would become rare, and the food supply would shrink"),
    ("you left milk out of the fridge for a week",
     "tiny bacteria would multiply in the milk, and it would sour, smell bad, and become unsafe to drink"),
    ("the moon disappeared tonight",
     "the tides would become much weaker, nights would be darker, and many animals that follow moonlight would be confused"),
    ("a tree was planted in a dark room",
     "the seedling would stretch thin and pale toward any light, and without sunlight it would soon stop making food and die"),
    ("everybody spoke the same language",
     "travel and trade would become simpler, but the world might slowly lose the poems, songs and ideas carried by older languages"),
    ("gravity became twice as strong",
     "everything would feel twice as heavy, jumping would be hard, and rain would fall much faster"),
    ("the sea level rose one meter",
     "many beaches would vanish, some coastal towns would flood, and people would have to build walls or move inland"),
    ("all the clocks stopped at noon",
     "people would still notice the sun moving across the sky, but trains, schools and shops would struggle to agree on the time"),
    ("a city planted a tree for every person",
     "the air would become cleaner, summers would feel cooler, and birds and insects would return to the streets"),
]

DEFINITIONS = [
    ("a fraction", "a way to show a part of a whole, like one half of a pie or three quarters of an hour"),
    ("an island", "a piece of land that has water on every side"),
    ("a shadow", "a dark shape made when something blocks the light"),
    ("a habit", "something a person does so often that it becomes almost automatic"),
    ("a drought", "a long time with little or no rain"),
    ("a village", "a small group of houses and shops, smaller than a town"),
    ("a harvest", "the gathering of ripe crops from the fields"),
    ("an apology", "words that say you are sorry for something you did"),
    ("a boundary", "a line or edge that shows where one thing ends and another begins"),
    ("a recipe", "a list of ingredients and steps for making a dish"),
    ("evaporation", "the slow changing of a liquid into a gas, the way a puddle dries up in the sun"),
    ("a peninsula", "a piece of land with water on most sides that is still joined to a larger land"),
    ("a fable", "a short story, often about animals, that teaches a lesson"),
    ("a compass", "a small tool with a needle that points north, used to find directions"),
    ("a habitat", "the kind of place where a plant or animal naturally lives"),
    ("gravity", "the pull that holds our feet on the ground and keeps the moon near the Earth"),
    ("a symptom", "a sign in the body that shows something is wrong, like a fever"),
    ("a proverb", "a short old saying that shares a piece of everyday wisdom"),
    ("an archive", "a place where old documents and records are kept safe"),
    ("erosion", "the slow wearing away of rock and soil by wind and water"),
    ("a summary", "a short retelling that keeps only the main points"),
    ("a guild", "a group of craftworkers who share skills and protect their trade"),
]

RECIPES = [
    ("a simple vegetable soup",
     ["2 carrots", "2 potatoes", "1 onion", "a handful of green beans", "1 liter of water", "salt", "a spoon of oil"],
     ["chop the onion, carrots and potatoes into small pieces.",
      "heat the oil in a pot and soften the onion for five minutes.",
      "add the carrots, potatoes, beans and water, and bring the pot to a boil.",
      "lower the heat, cover the pot, and let it simmer for twenty minutes.",
      "add salt to taste and serve the soup hot with bread."]),
    ("pancakes",
     ["1 cup of flour", "1 egg", "1 cup of milk", "a pinch of salt", "a spoon of sugar", "a little butter"],
     ["mix the flour, salt and sugar in a bowl.",
      "add the egg and milk and stir until the batter is smooth.",
      "melt a little butter in a pan over medium heat.",
      "pour in a small ladle of batter and cook until bubbles appear.",
      "flip the pancake and cook the other side until golden.",
      "serve warm with honey or fruit."]),
    ("a cheese sandwich",
     ["two slices of bread", "a slice of cheese", "a tomato", "a few leaves of lettuce", "butter"],
     ["spread a thin layer of butter on both slices of bread.",
      "wash the tomato and cut it into thin rounds.",
      "place the cheese, tomato and lettuce on one slice.",
      "cover with the second slice and press gently.",
      "cut the sandwich in half and eat it fresh."]),
    ("rice with peas",
     ["1 cup of rice", "2 cups of water", "a cup of peas", "salt", "a spoon of oil"],
     ["rinse the rice in cold water.",
      "boil the water with salt and a spoon of oil.",
      "add the rice, lower the heat, and cover the pot.",
      "ten minutes later, stir in the peas and cover again.",
      "when the water is gone and the rice is soft, fluff it with a fork and serve."]),
    ("a fruit salad",
     ["2 apples", "2 bananas", "an orange", "a handful of grapes", "a spoon of honey"],
     ["wash all the fruit.",
      "peel the bananas and the orange and cut them into pieces.",
      "cut the apples into bite-sized chunks.",
      "mix everything in a large bowl.",
      "drizzle honey over the top and chill before serving."]),
    ("scrambled eggs",
     ["2 eggs", "a pinch of salt", "a little butter", "a spoon of milk"],
     ["crack the eggs into a bowl, add the salt and milk, and beat well.",
      "melt the butter in a pan over low heat.",
      "pour in the eggs and stir slowly with a wooden spoon.",
      "when the eggs are softly set, take the pan off the heat and serve."]),
    ("baked apples",
     ["4 apples", "2 spoons of sugar", "a pinch of cinnamon", "a little butter"],
     ["wash the apples and cut out the cores.",
      "mix the sugar and cinnamon and fill the apples with it.",
      "put a small piece of butter on top of each apple.",
      "bake at medium heat for about twenty-five minutes, until soft.",
      "let them cool a little before serving."]),
    ("lemonade",
     ["2 lemons", "4 cups of cold water", "3 spoons of sugar", "some ice"],
     ["roll the lemons on the table to make them juicy.",
      "cut them in half and squeeze out the juice.",
      "stir the juice, water and sugar together until the sugar dissolves.",
      "add ice and a slice of lemon to each glass and serve cold."]),
]

PROCEDURES = [
    ("write a friendly letter",
     ["write the date in the top right corner.",
      "begin with a greeting such as Dear Anna,",
      "in the first paragraph, ask how the person is and share your news.",
      "in the second paragraph, write about the reason for the letter.",
      "close with warm wishes and your name at the bottom."]),
    ("plant bean seeds",
     ["fill a small pot with loose soil and make a hole two fingers deep.",
      "drop one seed into each hole and cover it gently with soil.",
      "water the pot until the soil is damp but not muddy.",
      "place the pot in a warm spot with daylight.",
      "keep the soil slightly wet and watch for the first green shoots in about a week."]),
    ("tidy a bedroom",
     ["open the window to let in fresh air.",
      "make the bed and put away anything on it.",
      "gather all the clothes and sort them into clean and dirty piles.",
      "return books, pens and small objects to their proper places.",
      "dust the shelves and finish by sweeping or vacuuming the floor."]),
    ("set a table for dinner",
     ["place a plate in front of every chair.",
      "put a fork on the left of each plate and a knife on the right.",
      "set a spoon to the right of the knife if there is soup.",
      "put a glass above the knife and a napkin beside the fork.",
      "check that everyone has a chair before serving the food."]),
    ("wash your hands properly",
     ["wet your hands with clean running water.",
      "rub soap over every part of your hands, including between the fingers.",
      "keep rubbing for about twenty seconds.",
      "rinse away all the soap under the water.",
      "dry your hands with a clean towel."]),
    ("pack a bag for a day trip",
     ["check the weather first so you know what to wear.",
      "pack a bottle of water and some food.",
      "add a hat, a warm layer and a small first aid kit.",
      "take a map or a charged phone.",
      "tell someone where you are going and when you plan to return."]),
    ("care for a houseplant",
     ["learn how much light your plant needs and place it well.",
      "touch the soil; water only when the top feels dry.",
      "remove yellow or dead leaves with clean scissors.",
      "wipe dust from the leaves so the plant can breathe.",
      "move it to a bigger pot when roots show at the bottom."]),
    ("prepare for an exam",
     ["find out exactly which topics the exam will cover.",
      "make a study plan that spreads the work over several days.",
      "study in short focused sessions with real breaks.",
      "explain the ideas out loud in your own words.",
      "sleep well the night before instead of studying all night."]),
    ("make a paper boat",
     ["take a rectangular sheet of paper and fold it in half.",
      "fold the two top corners down to the middle line.",
      "fold the bottom edge up on both sides.",
      "open the shape and flatten it into a square.",
      "fold the corners again, open it, and gently pull the top points apart to form the boat."]),
    ("clean a pair of shoes",
     ["knock the shoes together outside to remove loose dirt.",
      "brush off dried mud with an old brush.",
      "wipe the shoes with a damp cloth and a little soap.",
      "let them dry away from direct heat.",
      "finish with polish if the shoes are leather."]),
]

RULES = [
    ("the school library",
     ["speak quietly so others can read.",
      "return books on time.",
      "keep food and drinks away from the books.",
      "put books back where you found them.",
      "ask the librarian if you need help."]),
    ("the swimming pool",
     ["shower before entering the water.",
      "walk near the pool; never run.",
      "children must stay with an adult.",
      "no diving in the shallow end.",
      "listen to the lifeguard at all times."]),
    ("the community garden",
     ["close the gate behind you.",
      "water only your own plot unless asked.",
      "share tools and return them clean.",
      "do not pick other people's vegetables.",
      "keep the paths free of tools and hoses."]),
    ("playing catch in the yard",
     ["make sure everyone is ready before you throw.",
      "throw the ball so your partner can reach it.",
      "call out if the ball goes towards someone's face.",
      "keep the ball away from windows.",
      "stop playing when someone asks for the ball."]),
]

# --------------------------------------------------------------- helpers --

def _s(templates: List[str], rng: random.Random) -> str:
    return rng.choice(templates)


FAMILIES: Dict[str, Callable[[random.Random], str]] = {}


def family(name: str):
    def wrap(fn):
        FAMILIES[name] = fn
        return fn
    return wrap


# ------------------------------------------------------------ conversation --

@family("conv_plans")
def conv_plans(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    act = rng.choice(ACTIVITIES)
    place = rng.choice(PLACES)
    day = rng.choice(["Saturday", "Sunday", "Friday", "Thursday"])
    time_ = rng.choice(["ten", "nine", "three", "two", "eleven", "four"])
    return (
        f"{a}: Are you free this {day}?\n"
        f"{b}: I think so. Why do you ask?\n"
        f"{a}: I was hoping we could have {act} at {place}.\n"
        f"{b}: That sounds lovely. What time should we meet?\n"
        f"{a}: How about {time_} o'clock?\n"
        f"{b}: Perfect. Should I bring anything?\n"
        f"{a}: Just bring some water, and maybe a hat if it is {rng.choice(WEATHER)}.\n"
        f"{b}: All right, see you on {day} then.\n"
        f"{a}: See you! I am looking forward to it."
    )


@family("conv_favor")
def conv_favor(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    obj = rng.choice(OBJECTS)
    thanks = rng.choice(["Thanks so much, I owe you one.", "You're a lifesaver.",
                         "That's very kind of you.", "I really appreciate it."])
    return (
        f"{a}: Could you do me a favor?\n"
        f"{b}: Of course. What do you need?\n"
        f"{a}: I left my {obj} at {rng.choice(PLACES)} yesterday. Could you pick it up for me?\n"
        f"{b}: No problem. I am going that way after lunch anyway.\n"
        f"{a}: {thanks}\n"
        f"{b}: Don't mention it. You would do the same for me.\n"
        f"{a}: I really would. When do you think you will be back?\n"
        f"{b}: Around four. I will call you as soon as I have it."
    )


@family("conv_help_task")
def conv_help_task(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    task = rng.choice(["move these boxes", "paint the fence", "fix this shelf",
                       "carry the shopping", "water the garden", "clean the garage",
                       "set up the tables", "dig a new flower bed"])
    reason = rng.choice(["they are too heavy for me", "it will take all day alone",
                         "I do not know where to start", "my back is not great today"])
    return (
        f"{a}: {b}, could you help me {task}?\n"
        f"{b}: Sure. Why, what happened?\n"
        f"{a}: Well, {reason}.\n"
        f"{b}: Say no more. Give me ten minutes to finish my tea.\n"
        f"{a}: Take your time. We can start whenever you are ready.\n"
        f"{b}: All right. Where do we begin?\n"
        f"{a}: Let's start with the hardest part, and the rest will feel easy.\n"
        f"{b}: Good plan. I like working with you, {a}."
    )


@family("conv_advice")
def conv_advice(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    problem, advice = rng.choice([
        ("I keep forgetting new English words",
         "write a few words on small cards and look at them every morning"),
        ("I cannot sleep well lately",
         "stop looking at your phone an hour before bed and read instead"),
        ("I am always late in the morning",
         "prepare your clothes and bag the night before"),
        ("My plants keep dying",
         "water them less often and make sure the pots have holes at the bottom"),
        ("I get nervous speaking to new people",
         "start with small conversations, like talking to the shopkeeper"),
        ("I keep losing my keys",
         "put a bowl by the door and always drop them there"),
        ("I waste too much time on my phone",
         "keep the phone in another room when you need to work"),
        ("My bread never rises",
         "check that your yeast is fresh and the water is warm, not hot"),
        ("I can't keep my room tidy",
         "put one thing away every time you leave the room"),
    ])
    return (
        f"{a}: Can I ask you for some advice?\n"
        f"{b}: Of course, what is on your mind?\n"
        f"{a}: {problem}.\n"
        f"{b}: I see. Have you tried to {advice}?\n"
        f"{a}: No, I never thought of that.\n"
        f"{b}: It worked well for me. The trick is to do it every day, not just once.\n"
        f"{a}: That makes sense. I will try it for a week and see.\n"
        f"{b}: Good. Tell me how it goes, all right?"
    )


@family("conv_disagree")
def conv_disagree(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    thing = rng.choice(["the new schedule", "the music at the party", "the color of the walls",
                        "the plan for the trip", "the new carpet", "the movie last night",
                        "the restaurant downtown", "the team uniform", "the meeting time"])
    view_a, view_b = rng.choice([
        ("I think it is wonderful", "I honestly cannot stand it"),
        ("I really like it", "I am not a fan"),
        ("It seems fair to me", "It feels unfair to me"),
        ("It is a big improvement", "It was better before"),
        ("It is worth the money", "It is a waste of money"),
    ])
    return (
        f"{a}: So what do you think of {thing}?\n"
        f"{b}: {view_b}, to be honest. And you?\n"
        f"{a}: Really? {view_a}.\n"
        f"{b}: Huh. Why do you say that?\n"
        f"{a}: Because it makes everyday life easier. You will get used to it.\n"
        f"{b}: Maybe you are right. I just need more time.\n"
        f"{a}: That is fair. We do not have to agree about everything.\n"
        f"{b}: True. But I am glad you told me your side."
    )


@family("conv_smalltalk")
def conv_smalltalk(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    w = rng.choice(WEATHER)
    topic = rng.choice([
        ("did anything interesting happen today", "not much, just the usual busy day"),
        ("how is your family", "everyone is well, thanks for asking"),
        ("have you read anything good lately", f"a few pages of a book about {rng.choice(TOPICS_BOOKS)}"),
        ("are you still busy at work", "very busy, but it keeps the days moving"),
        ("how was your weekend", "quiet, which is exactly what I needed"),
    ])
    return (
        f"{a}: Good morning, {b}! Lovely to see you.\n"
        f"{b}: Morning, {a}. It is {w} today, isn't it?\n"
        f"{a}: It certainly is. So, {topic[0]}?\n"
        f"{b}: Oh, {topic[1]}.\n"
        f"{a}: That sounds about right. By the way, do you still go to {rng.choice(PLACES)} on Tuesdays?\n"
        f"{b}: Sometimes. It depends on the weather, honestly.\n"
        f"{a}: I understand that. Well, I should get going.\n"
        f"{b}: Me too. It was nice talking with you.\n"
        f"{a}: You too. Take care, {b}."
    )


@family("conv_illness")
def conv_illness(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    symptom = rng.choice(["a bad headache", "a sore throat", "an upset stomach",
                          "a stubborn cough", "a runny nose", "a stiff neck"])
    return (
        f"{a}: You don't look well today. Are you all right?\n"
        f"{b}: Not really. I have {symptom} since yesterday.\n"
        f"{a}: I'm sorry to hear that. Did you take anything for it?\n"
        f"{b}: Just some warm tea with honey so far.\n"
        f"{a}: You should rest and drink plenty of water.\n"
        f"{b}: That's what my mother says too.\n"
        f"{a}: Your mother is wise. If it gets worse, see a doctor.\n"
        f"{b}: I will. Hopefully it goes away by tomorrow.\n"
        f"{a}: Get well soon, {b}."
    )


@family("conv_shopping")
def conv_shopping(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    items = rng.sample(["eggs", "milk", "flour", "rice", "apples", "carrots", "cheese",
                        "bread", "potatoes", "onions", "honey", "lemons"], 3)
    return (
        f"{a}: Are you heading to the market?\n"
        f"{b}: Yes, in half an hour. Do you need anything?\n"
        f"{a}: If it's not too much trouble, could you get some {items[0]} and {items[1]}?\n"
        f"{b}: Of course. Anything else?\n"
        f"{a}: Maybe a bag of {items[2]}, if the price is fair.\n"
        f"{b}: I'll have a look. How much do you want to spend?\n"
        f"{a}: Nothing fancy, just the usual amount.\n"
        f"{b}: Understood. I'll see you when I get back.\n"
        f"{a}: Thank you so much, {b}. I really appreciate it."
    )


@family("conv_lost_item")
def conv_lost_item(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    obj = rng.choice(OBJECTS)
    place = rng.choice(PLACES)
    return (
        f"{a}: Have you seen my {obj}? I can't find it anywhere.\n"
        f"{b}: When did you last have it?\n"
        f"{a}: This morning, I think. I had it at {place}.\n"
        f"{b}: Let me think... Actually, I saw a {obj} on the shelf by the door.\n"
        f"{a}: That's it! I must have put it down when I came in.\n"
        f"{b}: Happens to everyone. You were in a hurry this morning.\n"
        f"{a}: I really was. Thanks, {b}. I was starting to worry.\n"
        f"{b}: No need to thank me. Maybe slow down a little tomorrow."
    )


@family("conv_weekend_review")
def conv_weekend_review(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    act1, act2 = rng.sample(ACTIVITIES, 2)
    feel = rng.choice(FEELINGS)
    return (
        f"{a}: How was your weekend, {b}?\n"
        f"{b}: Quite good. On Saturday we had {act1}, and on Sunday {act2}.\n"
        f"{a}: That sounds like a full weekend! How do you feel now?\n"
        f"{b}: Honestly, {feel}. And you?\n"
        f"{a}: Mine was quieter. I stayed home and cooked {rng.choice(FOODS)}.\n"
        f"{b}: That sounds restful. Sometimes a slow weekend is the best kind.\n"
        f"{a}: I agree. Anyway, the week is waiting for us.\n"
        f"{b}: It always is. Have a good Monday, {a}."
    )


@family("conv_teaching_moment")
def conv_teaching_moment(rng: random.Random) -> str:
    adult = rng.choice(["Grandma", "Grandpa", "Mom", "Dad", "Aunt Rosa", "Uncle Dan"])
    child = rng.choice(FIRST_NAMES)
    subj, why = rng.choice(WHY_SUBJECTS)
    return (
        f"{child}: {adult}, why {subj}?\n"
        f"{adult}: What a good question, little one.\n"
        f"{child}: Is it a long answer?\n"
        f"{adult}: Not too long. You see, {why}.\n"
        f"{child}: Oh! So that's why!\n"
        f"{adult}: Exactly. You learn something new every day.\n"
        f"{child}: Can I ask another question tomorrow?\n"
        f"{adult}: You can ask a hundred. That is how we grow wise, {child}."
    )


@family("conv_phone_invite")
def conv_phone_invite(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    event = rng.choice(["my birthday dinner", "a small garden party", "game night",
                        "a movie night", "dinner at my place", "a barbecue", "a picnic"])
    day = rng.choice(["Saturday", "Sunday"])
    return (
        f"{a}: Hello, is this {b}?\n"
        f"{b}: Speaking. Hi {a}, how are you?\n"
        f"{a}: Very well, thanks. Listen, I am calling to invite you to {event} this {day}.\n"
        f"{b}: Oh, how nice! What time does it start?\n"
        f"{a}: Around six. Nothing formal, just good food and good company.\n"
        f"{b}: I'd love to come. Can I bring something?\n"
        f"{a}: Just bring yourself. And maybe that wonderful salad of yours.\n"
        f"{b}: Deal. See you on {day}, {a}.\n"
        f"{a}: See you! I'm already looking forward to it."
    )


# --------------------------------------------------------------------- qa --

@family("qa_definition")
def qa_definition(rng: random.Random) -> str:
    term, definition = rng.choice(DEFINITIONS)
    q, a = rng.choice(FIRST_NAMES[::7]), rng.choice(FIRST_NAMES[1::7])
    q_form = rng.choice([f"What is {term}?", f"Can you tell me what {term} is?",
                         f"What does '{term}' mean?", f"What exactly is {term}?"])
    a_lead = rng.choice(["Here is a simple answer.", "Good question.",
                         "Let me explain.", "That one is easy."])
    return (f"{q}: {q_form}\n"
            f"{a}: {a_lead} To put it simply, {term} is {definition}. "
            f"That is really all there is to it, though of course the details can grow deeper.")


@family("qa_why")
def qa_why(rng: random.Random) -> str:
    subj, why = rng.choice(WHY_SUBJECTS)
    q = rng.choice(FIRST_NAMES[2::7])
    a = rng.choice(FIRST_NAMES[3::7])
    q_form = rng.choice([f"Why {subj}?", f"Why is it that {subj}?",
                         f"Can you explain why {subj}?"])
    follow = rng.choice([
        "It is one of those everyday things you never stop to think about.",
        "Once you hear it, it seems obvious, doesn't it?",
        "Nature has quiet reasons for everything.",
        "You can see this for yourself the next time you pay attention.",
    ])
    return (f"{q}: {q_form}\n"
            f"{a}: The short answer is that {why}.\n"
            f"{q}: Really? I never knew that.\n"
            f"{a}: {follow}")


@family("qa_how")
def qa_how(rng: random.Random) -> str:
    subj, how = rng.choice(HOW_SUBJECTS)
    q = rng.choice(FIRST_NAMES[4::7])
    a = rng.choice(FIRST_NAMES[5::7])
    q_form = rng.choice([f"How does it happen that {subj}?",
                         f"Can you explain how {subj}?",
                         f"Do you know how {subj}?"])
    return (f"{q}: {q_form}\n"
            f"{a}: It works like this: {how}.\n"
            f"{q}: That is simpler than I expected.\n"
            f"{a}: The best explanations usually are.")


@family("qa_diff")
def qa_diff(rng: random.Random) -> str:
    x, y, diff = rng.choice(DIFF_PAIRS)
    q = rng.choice(FIRST_NAMES[6::7])
    a = rng.choice(FIRST_NAMES[::8])
    return (f"{q}: What is the difference between {x} and {y}?\n"
            f"{a}: People mix these up all the time. The difference is that {diff}.\n"
            f"{q}: Ah, so they are not the same thing at all.\n"
            f"{a}: Not quite, no. Now you will never confuse them again.")


@family("qa_what_if")
def qa_what_if(rng: random.Random) -> str:
    cond, result = rng.choice(WHAT_IF)
    q = rng.choice(FIRST_NAMES[1::8])
    a = rng.choice(FIRST_NAMES[2::8])
    return (f"{q}: What would happen if {cond}?\n"
            f"{a}: That is a serious thought. Most likely, {result}.\n"
            f"{q}: I hope it never happens.\n"
            f"{a}: So do I, but it is good to imagine these things. It teaches us what to value.")


@family("qa_where")
def qa_where(rng: random.Random) -> str:
    place, desc = rng.choice([
        ("the Sahara", "in northern Africa, stretching across many countries"),
        ("the Amazon river", "in South America, mostly crossing Brazil"),
        ("Mount Everest", "in Asia, on the border of Nepal and China"),
        ("the Great Barrier Reef", "in the ocean near the northeast coast of Australia"),
        ("Paris", "in France, in the northern part of the country"),
        ("the Nile", "in northeastern Africa, flowing north to the sea"),
        ("Antarctica", "at the south end of the Earth, the coldest continent"),
        ("the Andes", "along the western side of South America"),
        ("Tokyo", "in Japan, on the eastern coast of the biggest island"),
        ("the Mediterranean Sea", "between southern Europe and northern Africa"),
    ])
    q = rng.choice(FIRST_NAMES[3::8])
    a = rng.choice(FIRST_NAMES[4::8])
    return (f"{q}: Where is {place}?\n"
            f"{a}: {place.capitalize()} is {desc}.\n"
            f"{q}: I'd like to see it one day.\n"
            f"{a}: Reading about it is a good first step.")


@family("qa_when")
def qa_when(rng: random.Random) -> str:
    qtext, ans = rng.choice([
        ("When do swallows return in the spring?", "when the days grow warm and insects fill the air again"),
        ("When is the best time to plant tomatoes?", "after the last frost, when the soil has warmed up"),
        ("When did people first sail across the ocean?", "thousands of years ago, long before engines, with only wind and courage"),
        ("When should you water flowers?", "early in the morning or in the evening, when the sun is not too strong"),
        ("When does the moon look full?", "about once a month, when the whole face we see is lit by the sun"),
        ("When is the shortest day of the year?", "in the middle of winter, at the winter solstice"),
    ])
    q = rng.choice(FIRST_NAMES[5::8])
    a = rng.choice(FIRST_NAMES[6::8])
    return (f"{q}: {qtext}\n"
            f"{a}: The answer is {ans}.\n"
            f"{q}: That is useful to know. Thank you.\n"
            f"{a}: Any time. Asking questions is how we learn.")


@family("qa_who")
def qa_who(rng: random.Random) -> str:
    qtext, ans = rng.choice([
        ("Who wrote the play Romeo and Juliet?", "William Shakespeare, an English playwright"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci, an Italian artist"),
        ("Who was the first person to walk on the moon?", "Neil Armstrong, an American astronaut, in 1969"),
        ("Who wrote the fairy tales about the little mermaid?", "Hans Christian Andersen, a Danish writer"),
        ("Who discovered gravity's laws with the falling apple story?", "Isaac Newton, an English scientist"),
        ("Who wrote the book about the wooden boy Pinocchio?", "Carlo Collodi, an Italian author"),
    ])
    q = rng.choice(FIRST_NAMES[::9])
    a = rng.choice(FIRST_NAMES[1::9])
    return (f"{q}: {qtext}\n"
            f"{a}: That was {ans}.\n"
            f"{q}: I should remember that for the quiz.\n"
            f"{a}: You should indeed.")


@family("qa_everyday_best")
def qa_everyday_best(rng: random.Random) -> str:
    qtext, ans, why = rng.choice([
        ("What is the best way to learn a new language?", "a little practice every day",
         "short daily practice builds memory better than long weekly sessions"),
        ("How should you store fresh bread?", "in a paper bag or a bread box, not in plastic",
         "plastic traps moisture and makes the crust soft and moldy"),
        ("What is the safest way to carry a knife in the kitchen?", "pointing down at your side",
         "if you trip, the blade points away from your body"),
        ("How do you stop hiccups?", "sip cold water slowly and breathe calmly",
         "slow swallowing calms the muscle that causes the hiccup"),
        ("What should you do if you spill water on the floor?", "dry it right away",
         "a wet floor is slippery and someone could fall"),
        ("How can you tell if an egg is fresh?", "put it in water; a fresh egg sinks",
         "old eggs fill with air over time and start to float"),
    ])
    q = rng.choice(FIRST_NAMES[2::9])
    a = rng.choice(FIRST_NAMES[3::9])
    return (f"{q}: {qtext}\n"
            f"{a}: The best answer is {ans}.\n"
            f"{q}: Why is that?\n"
            f"{a}: Because {why}.")


# -------------------------------------------------------------- description --

@family("desc_room")
def desc_room(rng: random.Random) -> str:
    items = rng.sample(ROOM_ITEMS, 3)
    light = rng.choice(["soft morning light", "the warm glow of a lamp",
                        "pale afternoon light", "the orange light of the fire",
                        "thin gray light from a small window"])
    mood = rng.choice(["peaceful", "slightly messy but welcoming", "old-fashioned and tidy",
                       "quiet and cool", "warm and full of good smells"])
    return (f"The room felt {mood}. {light.capitalize()} fell across the floor. "
            f"In one corner stood {items[0]}, and along the far wall there was {items[1]}. "
            f"Near the door, {items[2]} waited as if it had always been there. "
            f"It was the kind of room where you could sit for an hour and not notice the time.")


@family("desc_person")
def desc_person(rng: random.Random) -> str:
    name = rng.choice(FIRST_NAMES)
    prof = rng.choice(PROFESSIONS)
    feat = rng.choice(["gentle eyes", "a slow, careful way of speaking", "working hands with rough skin",
                       "a fringe of gray hair", "a warm, easy laugh", "a straight back and quick step",
                       "a soft voice that made people listen"])
    habit = rng.choice(["always carried a small notebook", "never rushed anyone",
                        "hummed quietly while working", "remembered everyone's name",
                        "fed the birds every morning", "kept candy in every pocket",
                        "read the newspaper cover to cover"])
    return (f"{name} was a {prof} with {feat}. "
            f"People in the town knew {name} well, because {name.lower() if False else name} {habit}. "
            f"Years of {rng.choice(['patience', 'hard work', 'early mornings', 'careful attention'])} showed in "
            f"everything {name} did, and strangers trusted them within minutes.")


@family("desc_landscape")
def desc_landscape(rng: random.Random) -> str:
    kind = rng.choice([
        ("mountains", "peaks hidden in moving clouds", "the air grew thin and cold"),
        ("forest", "tall pines that whispered when the wind moved", "the ground was soft with old needles"),
        ("coast", "white waves breaking on dark rocks", "the wind tasted of salt"),
        ("farmland", "fields striped green and gold", "a farmer's dog barked far away"),
        ("river valley", "a slow river bending through meadows", "herons stood still in the shallows"),
        ("marsh", "pools of still water between the reeds", "frogs called to each other in the dusk"),
    ])
    w = rng.choice(WEATHER)
    return (f"The {kind[0]} stretched out in every direction, with {kind[1]}. "
            f"That day it was {w}, and {kind[2]}. "
            f"From a distance the land looked empty, but the longer you watched, "
            f"the more life you noticed: birds working the edges of the wind, "
            f"small things moving through the grass, water finding its way downhill.")


@family("desc_weather_scene")
def desc_weather_scene(rng: random.Random) -> str:
    scene = rng.choice([
        ("thunderstorm", "The sky darkened until afternoon looked like evening. "
         "Thunder rolled over the rooftops, and rain came down so hard the gutters sang."),
        ("first snow", "Snow began before dawn, soft flakes drifting past the window. "
         "By breakfast the whole street had turned white, and the usual noise of the town was gone."),
        ("heat wave", "By noon the heat lay over the town like a heavy blanket. "
         "The pavement shimmered, dogs slept in doorways, and even the birds were quiet."),
        ("autumn wind", "A restless wind pushed through the streets all day, "
         "pulling red and gold leaves from the trees and piling them against the fences."),
        ("spring rain", "Warm rain fell gently for hours, tapping on the leaves. "
         "Everything it touched seemed to turn a brighter green before your eyes."),
        ("morning fog", "Fog filled the streets at sunrise, thick enough to swallow houses whole. "
         "Lamps glowed like small moons, and footsteps arrived before the people did."),
    ])
    return scene[1] + " " + rng.choice([
        "People hurried along with their collars up.",
        "It was the kind of weather that made home feel extra welcoming.",
        "Inside, someone put a kettle on and watched it through the window.",
        "The town seemed to hold its breath and wait for it to pass.",
    ])


@family("desc_object")
def desc_object(rng: random.Random) -> str:
    obj = rng.choice(OBJECTS)
    age = rng.choice(["older than anyone could remember", "a gift from long ago",
                      "smooth from years of use", "stained but well cared for",
                      "scratched at the edges"])
    place = rng.choice(["on the highest shelf", "near the kitchen window",
                        "in the middle of the old desk", "beside the front door",
                        "on a hook by the stove"])
    return (f"The {obj} was {age}. It sat {place}, where it caught the light in the mornings. "
            f"Everyone who visited noticed it at once, though nobody could say why. "
            f"Perhaps it was the way it seemed to carry its history quietly, "
            f"like an old photograph that keeps its secrets.")


@family("desc_animal")
def desc_animal(rng: random.Random) -> str:
    an = rng.choice(ANIMALS)
    look = rng.choice(["quick eyes", "a cautious way of moving", "soft fur", "bright feathers",
                       "patient eyes", "a proud way of holding its head"])
    act = rng.choice(["watched the garden fence", "searched for food along the wall",
                      "sat perfectly still in the sun", "moved carefully through the grass",
                      "drank from the puddle by the gate"])
    return (f"The {an} had {look} and knew the yard better than anyone. "
            f"Every morning it {act}, unbothered by the people at the window. "
            f"It belonged to no one and to the whole street, "
            f"and the street was quietly proud of it.")


@family("desc_market")
def desc_market(rng: random.Random) -> str:
    goods1, goods2, goods3 = rng.sample(["fruit", "flowers", "cheese", "bread", "cloth",
                                         "pottery", "honey", "fish", "vegetables"], 3)
    return (f"The market was already busy by eight o'clock. Stalls sold {goods1}, {goods2} and {goods3}, "
            f"each seller calling out in a different rhythm. The air smelled of coffee and crushed herbs. "
            f"Shoppers moved slowly, stopping to taste, to argue about prices, to greet old friends. "
            f"By ten the square was so full you had to turn sideways to pass, "
            f"and nobody seemed to mind at all.")


# ------------------------------------------------------------ instruction --

@family("instr_recipe")
def instr_recipe(rng: random.Random) -> str:
    name, ingredients, steps = rng.choice(RECIPES)
    lines = [f"{rng.choice(['How to make', 'Recipe for', 'Making'])} {name}."]
    lines.append("You will need: " + ", ".join(ingredients[:-1]) + f", and {ingredients[-1]}.")
    lines.append("Steps:")
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}. {s[0].upper()}{s[1:]}")
    lines.append(rng.choice(["Enjoy it while it is fresh.", "Serve it with a smile.",
                             "That is all there is to it.", "Simple, honest food."]))
    return "\n".join(lines)


@family("instr_procedure")
def instr_procedure(rng: random.Random) -> str:
    name, steps = rng.choice(PROCEDURES)
    lines = [f"Here is how to {name}, step by step."]
    for i, s in enumerate(steps, 1):
        connector = ["First", "Then", "Next", "After that", "Finally"][min(i - 1, 4)] \
            if len(steps) > 3 else ["First", "Next", "Finally"][min(i - 1, 2)]
        lines.append(f"{i}. When you {name}," if i == 1 and rng.random() < 0.25
                     else f"{i}. {connector}, {s}")
    lines.append(rng.choice(["Do these steps in order and you will not go wrong.",
                             "With a little practice this becomes easy.",
                             "Take your time — careful steps beat fast ones."]))
    return "\n".join(lines)


@family("instr_rules")
def instr_rules(rng: random.Random) -> str:
    place, rules = rng.choice(RULES)
    lines = [f"Rules for {place.capitalize()}:"]
    lines.append(f"Everyone who uses {place} should follow these simple rules:")
    for i, r in enumerate(rules, 1):
        lines.append(f"{i}. {r[0].upper()}{r[1:]}")
    lines.append(rng.choice(["These rules keep the place pleasant for everyone.",
                             "Thank you for helping keep it a good place.",
                             "If in doubt, be kind and use common sense."]))
    return "\n".join(lines)


@family("instr_directions")
def instr_directions(rng: random.Random) -> str:
    a, b = rng.sample(["the post office", "the library", "the bakery", "the bus station",
                       "the school", "the pharmacy", "the museum"], 2)
    l1 = rng.choice(["turn left at the fountain", "walk past the church",
                     "cross the small stone bridge", "go straight through the market square"])
    l2 = rng.choice(["until you see a tall oak tree", "for about five minutes",
                     "until the road bends to the right", "past two narrow streets"])
    q = rng.choice(FIRST_NAMES)
    return (f"{q}: Excuse me, how do I get from {a} to {b}?\n"
            f"Local: Easy! Leave {a}, {l1}, then keep walking {l2}. "
            f"You will see {b} on your left; you cannot miss it.\n"
            f"{q}: Is it far?\n"
            f"Local: Not at all — a pleasant walk of ten minutes at most. "
            f"If you reach the river, you have gone too far.\n"
            f"{q}: Thank you very much!\n"
            f"Local: Any time. Enjoy the walk.")


# ------------------------------------------------------------ explanation --

@family("expl_why_long")
def expl_why_long(rng: random.Random) -> str:
    subj, why = rng.choice(WHY_SUBJECTS)
    opener = rng.choice([
        f"Have you ever wondered why {subj}?",
        f"People often ask why {subj}.",
        f"At first it seems strange that {subj}.",
        f"It is not obvious why {subj}, but there is a clear reason.",
    ])
    closer = rng.choice([
        "And once you notice it, you will see it everywhere.",
        "Nature rarely does anything without a reason.",
        "A small detail explains a very big question.",
        "Simple causes can have surprisingly large effects.",
    ])
    return (f"{opener} The explanation is that {why}. "
            f"It helps to picture it step by step rather than all at once. {closer}")


@family("expl_how_long")
def expl_how_long(rng: random.Random) -> str:
    subj, how = rng.choice(HOW_SUBJECTS)
    example = rng.choice([
        "You can watch this happen at home if you look closely.",
        "Once you know the steps, the whole thing looks simple.",
        "The same idea appears in many other places in everyday life.",
        "Engineers use this same principle in larger machines too.",
    ])
    return (f"Let me explain how {subj}. In plain words: {how}. "
            f"Each part depends on the one before it, like links in a chain. {example}")


@family("expl_definition_long")
def expl_definition_long(rng: random.Random) -> str:
    term, definition = rng.choice(DEFINITIONS)
    use = rng.choice([
        "You will meet this word in books, in school, and in everyday talk.",
        "It is a small word for a big idea.",
        "The word sounds formal, but the idea behind it is simple.",
        "Knowing this word makes many explanations easier to follow.",
    ])
    return (f"In simple English, {term} is {definition}. {use} "
            f"If you forget the details, hold on to this short meaning and the rest will come back.")


@family("expl_compare")
def expl_compare(rng: random.Random) -> str:
    x, y, diff = rng.choice(DIFF_PAIRS)
    return (f"People sometimes confuse {x} with {y}, but they are easy to tell apart "
            f"once you know the key idea. Simply put, {diff}. "
            f"Keep that one difference in mind, and you will use both words correctly.")


@family("expl_cause_chain")
def expl_cause_chain(rng: random.Random) -> str:
    chain = rng.choice([
        ("a seed becomes a tree", "first the seed drinks water and swells, then a small root pushes downward, "
         "then a green shoot climbs toward the light, and finally leaves open and begin to feed the young plant"),
        ("a cold becomes a fever", "first a small virus enters the body, then the body raises its temperature "
         "on purpose to slow the virus down, and that rise in temperature is what we call a fever"),
        ("batter rises into bread", "the yeast eats a little sugar, makes tiny bubbles of gas, "
         "the soft dough traps the bubbles, and the heat of the oven sets the risen shape in place forever"),
        ("a rumor spreads through a town", "one person tells two friends, each friend tells two more, "
         "and by evening the news has crossed the whole town, changing a little at every step"),
        ("a puddle disappears", "the sun warms the water, the top layer slowly turns into invisible vapor, "
         "and molecule by molecule the puddle rises into the air until nothing is left but dry ground"),
        ("an iron nail rusts", "water and oxygen work on the metal together, a reddish layer forms on the surface, "
         "and given enough time the whole nail crumbles into brown dust"),
    ])
    return (f"Here is how {chain[0]}, step by step: {chain[1]}. "
            f"Each step causes the next, which is why the process cannot be hurried. "
            f"Watching for cause and effect like this is the heart of clear thinking.")


@family("expl_math_words")
def expl_math_words(rng: random.Random) -> str:
    topic = rng.choice([
        ("what a half really means", "if you split something into two equal parts, each part is a half; "
         "two halves always make exactly one whole, no more and no less"),
        ("why zero is important", "zero lets us write numbers like ten and one hundred without inventing new signs, "
         "and it gives the idea of 'nothing' its own place in mathematics"),
        ("what multiplication is for", "instead of adding the same number many times, multiplication does it in one step; "
         "three groups of four apples is twelve apples, counted the quick way"),
        ("the difference between perimeter and area", "perimeter is the length of the fence around a field, "
         "while area is the amount of grass inside it; one is a length, the other is a surface"),
        ("why estimating helps", "a rough answer tells you whether your exact answer makes sense; "
         "if your estimate is about fifty and your answer is five thousand, something went wrong"),
        ("what an average is", "it is the fair share: if every person had exactly the same amount, "
         "that amount is the average, even if in reality nobody has exactly that much"),
    ])
    return (f"A short lesson about {topic[0]}. Imagine it in your head: {topic[1]}. "
            f"That is really the whole idea; the rest is practice.")


# ----------------------------------------------------------------- formal --

@family("formal_letter")
def formal_letter(rng: random.Random) -> str:
    a = rng.choice(FIRST_NAMES)
    surname = rng.choice(["Thompson", "Reyes", "Kowalski", "Bennett", "Okafor",
                          "Lindqvist", "Marsh", "Delgado", "Hartmann", "Novak"])
    reason, request = rng.choice([
        ("the fence between our properties has been damaged by the recent storm",
         "arrange an inspection at your earliest convenience"),
        ("my order from three weeks ago has not yet arrived",
         "look into the delay and inform me of the new delivery date"),
        ("the streetlight outside number fourteen has not worked for a month",
         "schedule a repair as soon as possible"),
        ("I will be unable to attend the meeting on Thursday due to a family matter",
         "receive the minutes afterward"),
        ("the water pressure in our street has been very low this week",
         "investigate the cause and restore normal service"),
    ])
    return (f"Dear Mr. {surname},\n\n"
            f"I am writing to inform you that {reason}. I would be grateful if you could "
            f"{request}.\n\n"
            f"Thank you for your attention to this matter. I look forward to your reply.\n\n"
            f"Yours sincerely,\n{a} {rng.choice(['Bell','Crane','Ford','Lennox','Ibarra','Whitfield'])}")


@family("formal_notice")
def formal_notice(rng: random.Random) -> str:
    place = rng.choice(["The community hall", "The public library", "The town museum",
                        "The swimming pool", "The market hall"])
    when = rng.choice(["Monday, the fourth of May", "Friday, the twelfth of June",
                       "Wednesday, the first of October"])
    reason = rng.choice(["scheduled maintenance", "the installation of new lighting",
                         "deep cleaning", "staff training", "repairs to the roof"])
    return (f"Notice to all visitors. {place} will be closed on {when} due to {reason}. "
            f"We apologize for any inconvenience and thank you for your understanding. "
            f"Normal opening hours will resume the following day. "
            f"For questions, please contact the office during working hours.")


@family("formal_report")
def formal_report(rng: random.Random) -> str:
    topic = rng.choice(["the town's water supply", "the new reading program",
                        "traffic near the school", "the condition of the old bridge",
                        "the community garden project"])
    finding1, finding2 = rng.choice([
        ("usage rose steadily throughout the summer", "the pipes will need attention within five years"),
        ("attendance doubled after the second month", "the largest group of readers was under ten years old"),
        ("the morning rush begins earlier than expected", "most problems occur at the two central crossings"),
        ("the stone supports remain sound", "the wooden surface should be replaced"),
        ("every plot was claimed within two weeks", "watering is the most common source of disagreement"),
    ])
    return (f"This report examines {topic}. Two findings are worth attention. "
            f"First, {finding1}. Second, {finding2}. "
            f"Based on these findings, the committee recommends a careful review before next summer, "
            f"followed by practical measures where the need is greatest. "
            f"Details and measurements are attached to the end of this document.")


# ------------------------------------------------------------- registration --

# ------------------------------------------------------------------ stories --
# Short original narratives. With public-domain downloads unavailable, the
# story slice of the corpus must come from authored text; these families are
# designed for surface variety: different frames, names, places, objects,
# weather and outcomes per sample.

STORY_FRAMES = [
    "lost_and_found", "first_day", "small_kindness", "weather_trouble",
    "the_fix", "the_mistake", "neighbors", "the_journey",
]

@family("story_found_object")
def story_found_object(rng: random.Random) -> str:
    name = rng.choice(FIRST_NAMES)
    place = rng.choice(PLACES)
    obj = rng.choice(["a brass key", "a leather notebook", "an old photograph",
                      "a silver thimble", "a wooden whistle", "a folded map",
                      "a tin of coins", "a child's red mitten", "a train ticket",
                      "a fountain pen", "a small carved bird", "a paper boat"])
    where = rng.choice(["under the third stair", "behind the radiator",
                        "in the pocket of a winter coat", "beneath the floor mat",
                        "inside a library book", "at the bottom of the tool box",
                        "in the hollow of the garden wall", "under the loose board",
                        "between the cushions of the old bench", "in the hen house"])
    weather = rng.choice(WEATHER)
    feeling = rng.choice(FEELINGS)
    keep_days = rng.choice(["three days", "a week", "a fortnight", "the whole winter"])
    owner = rng.choice(["the old stationmaster", "the baker's daughter",
                        "a traveling salesman", "the schoolteacher",
                        "the seamstress from the second floor", "the ferryman",
                        "a choir boy with freckles", "the midwife",
                        "the night porter at the hotel", "the gardener"])
    craft = rng.choice(PROFESSIONS)
    openers = [
        f"On a {weather} morning, not far from {place}, {name} found {obj} {where}.",
        f"It was {weather} when {name}, who worked as a {craft}, discovered {obj} {where}.",
        f"Nobody who frequented {place} could remember losing {obj}, but there it was, {where}, and {name} was the one who found it.",
    ]
    middles = [
        f"{name} turned it over slowly. It was worn at the edges, the way things get when they have been carried for years and loved without ceremony.",
        f"For a moment {name} simply stood and looked at it. There are things that arrive quietly and change the furniture of a day.",
        f"It was not valuable, anyone could see that. But it had the particular patience of an object that had been waiting on purpose.",
    ]
    waits = [
        f"{name} kept it for {keep_days}, asking after an owner in the least dramatic way: a word here, a word there, nothing pinned to a door.",
        f"For {keep_days} the thing lay on the kitchen table while {name} thought about what to do and made tea instead.",
        f"{name} waited {keep_days}. No one came. The shops did not gossip about it. The thing belonged to nobody, and for that reason it belonged to everywhere.",
    ]
    ends = [
        f"In the end it was {owner} who said, quietly, that they had stopped looking years ago. {name} handed it over without a speech. That, in {place}, was considered the entire reward: a {feeling} walk home and a story told twice at supper.",
        f"When {owner} finally recognized it, the whole thing took less than a minute. \"I wondered where you had gone,\" {owner} said to the object, not to {name}, which {name} thought exactly right.",
        f"{name} gave it to {owner} on a {weather} afternoon, and was offered money, and refused it, and was offered cake, and accepted. The {feeling} aftertaste of the afternoon lasted longer than the cake.",
    ]
    return f"{_s(openers, rng)}\n\n{_s(middles, rng)}\n\n{_s(waits, rng)}\n\n{_s(ends, rng)}"


@family("story_two_friends")
def story_two_friends(rng: random.Random) -> str:
    a, b = rng.sample(FIRST_NAMES, 2)
    place = rng.choice(PLACES)
    activity = rng.choice(ACTIVITIES)
    job_a = rng.choice(PROFESSIONS)
    food = rng.choice(FOODS)
    season = rng.choice(["spring", "high summer", "early autumn", "the deep of winter"])
    habit = rng.choice([
        "met every Friday without fail and never once said why Fridays",
        "argued about everything and agreed that this was the point",
        "had known each other so long that silence had become a shared language",
        "kept a running tally of small debts neither of them intended to settle",
        "never telephoned, because telephones made everything sound worse than it was",
    ])
    quirk_b = rng.choice([
        "arrived everywhere eleven minutes early and denied it",
        "carried an umbrella in all forecasts and lent it without conditions",
        "could name every bird by its song and nobody believed her",
        "remembered everyone's birthday and pretended not to",
        "saved good news for last the way other people save dessert",
    ])
    event = rng.choice([
        f"One {season} afternoon, {b} did not come. This had never happened, not once, and {a} sat with the {food} going cold and felt the first honest worry of a comfortable life.",
        f"One morning in {season}, {a} found a note in the usual place: three lines, untidy, unmistakably from {b}. It asked for help with something small and embarrassing.",
        f"On the first cold morning of {season}, {b} arrived with two parcels and a plan that required {a} to take a day off work, which was out of the question and therefore agreed immediately.",
    ])
    resolve = rng.choice([
        f"It turned out to be nothing, which is to say it turned out to be everything: {b} the {job_a} had simply forgotten the day of the week. They laughed about it for years, but {a} never laughed entirely, and brought two chairs out every Friday after, just in case either of them forgot again.",
        f"They {activity} until the light went, and fixed the small embarrassing thing, and never mentioned it again, which in {place} passes for a monument.",
        f"By evening they had done half the plan and eaten all the {food}, and agreed that the other half could wait another year. It did not wait, in the end; but that, as {b} said at the time, is what next year is for.",
    ])
    return (f"{a} and {b} had been friends since they were children; everyone "
            f"around {place} knew the two of them by sight. "
            f"They {habit}. {a} was steady and {b} {quirk_b}.\n\n{event}\n\n{resolve}")


@family("story_weather_day")
def story_weather_day(rng: random.Random) -> str:
    place = rng.choice(PLACES)
    person = rng.choice(FIRST_NAMES)
    job = rng.choice(PROFESSIONS)
    storm = rng.choice(["the great storm", "the week of fog", "the sudden frost",
                        "the three-day rain", "the heatwave", "the freak hailstorm",
                        "the long drought", "the blizzard"])
    prep = rng.choice([
        "boarded the windows against the wind",
        "stacked sandbags along the garden wall",
        "carried the beehives into the lee of the barn",
        "wrapped the water pipes in old blankets",
        "moved the boats to the upper moorings",
        "covered the seedbeds with every sheet in the parish",
    ])
    small = rng.choice([
        "a last letter delivered by hand",
        "a kettle passed over a fence",
        "a lost dog returned to the grocer",
        "an extra candle in every window",
        "a pot of soup that toured four kitchens in one evening",
        "a borrowed umbrella walked home twice",
    ])
    after = rng.choice([
        "and the town counted roofs like a prayer and found every one accounted for",
        "and when it was over the streets smelled of clean beginnings",
        "and the lost things came back in ones and twos, the way cats come home",
        "and the schoolmaster said it had been the best-attended geography lesson in living memory",
        "and for months afterwards people finished the story with the same sentence: we were ready, and we were lucky, and we were both",
    ])
    return (f"Nobody who lived near {place} at the time forgot {storm}. "
            f"{person} the {job} remembered it best, because before it came the whole town "
            f"{prep}.\n\nThe small thing {person} remembered was not the wind or the damage. "
            f"It was {small}, which seemed afterwards to have been the point of the whole exercise: "
            f"{after}.")


@family("story_first_job")
def story_first_job(rng: random.Random) -> str:
    person = rng.choice(FIRST_NAMES)
    job = rng.choice(["baker's boy", "lighthouse keeper's assistant", "grocer",
                      "apprentice tailor", "stable hand", "telephone operator",
                      "milkman", "ticket clerk", "park keeper", "printer's devil"])
    place = rng.choice(PLACES)
    elder = rng.choice(["the old master", "the head keeper", "the forewoman",
                        "the senior clerk", "the chief engineer"])
    advice = rng.choice([
        "Start before the bell and nobody will ever ask what time you arrive.",
        "The customer is not always right, but the customer is always there.",
        "Learn the names first; the rest of the job will introduce itself.",
        "Everything is heavy until you learn to carry it; then it is only yours.",
        "Keep the sharp tools sharp and the blunt opinions blunter.",
    ])
    mistake = rng.choice([
        f"On the first morning, {person} dropped, broke, mislaid or misdirected almost everything that could be dropped, broken, mislaid or misdirected.",
        f"{person} made exactly one mistake in the first week, but it was a thorough one, involving the wrong delivery, the wrong address, and the wrong side of town on market day.",
        f"Nothing went wrong until the third day, when everything went wrong at once, including, for reasons nobody explained, a goat.",
    ])
    keepers = rng.choice([
        "kept the job for forty-one years and sent postcards to the same address long after it stopped existing",
        "left after two winters but used the lesson every working day afterwards",
        "grew into the job the way a tree grows into a fence, until nobody could say where one ended and the other began",
        "taught the next apprentice the same first sentence, and the one after that said the same thing, which is how a place keeps its mind",
    ])
    return (f"{person}'s first job was as a {job} in {place}. The {elder} offered, "
            f"on the first morning, exactly one piece of advice: \"{advice}\"\n\n"
            f"{mistake}\n\nIt was the advice that survived the mistake, and the mistake "
            f"that made the advice worth keeping. {person} {keepers}.")


@family("story_small_kindness")
def story_small_kindness(rng: random.Random) -> str:
    giver = rng.choice(FIRST_NAMES)
    receiver = rng.choice(FIRST_NAMES)
    place = rng.choice(PLACES)
    weather = rng.choice(WEATHER)
    thing = rng.choice(["a seat on the crowded bus", "the last umbrella in the shop",
                        "a packet of sandwiches", "directions walked, not pointed",
                        "a window held open on the train", "the good ladder",
                        "a lift to the early ferry", "a bag of ripe plums",
                        "the warm pair of gloves", "ten minutes of patient listening"])
    reason = rng.choice([
        "was tired in the way that sleep does not fix",
        "had forgotten the day of the week and with it the shopping money",
        "was new in town and apologizing to stationary objects",
        "had just received bad news in a very good hat",
        "was carrying something heavier than it looked",
    ])
    echo = rng.choice([
        "Years later, neither of them could remember what had been said, only what had been done, which they both agreed was the correct ratio.",
        "It was nothing, both of them said later, and both of them meant something by it.",
        "The debt was never mentioned again, which is how some debts get paid with interest.",
        "A week later the same thing happened in reverse, with different weather, which both of them found quietly suspicious and did not discuss.",
    ])
    return (f"On a {weather} afternoon in {place}, {receiver} {reason}. "
            f"{giver}, a stranger at the time, offered {thing}.\n\n"
            f"There was no speech. There were no witnesses except the ticket collector, "
            f"who told the story later in two sentences and improved neither. "
            f"{echo}")


@family("story_animal_day")
def story_animal_day(rng: random.Random) -> str:
    animal = rng.choice(["the ginger cat", "the one-eyed magpie", "the old donkey",
                         "the harbor seal", "the stationmaster's dog", "the lame goose",
                         "the butcher's parrot", "the churchyard fox"])
    place = rng.choice(PLACES)
    habit = rng.choice([
        "inspected everything that arrived by truck and approved of none of it",
        "slept through sermons and woke for picnics",
        "demanded tribute at the same three doors in the same order",
        "escorted children to the corner and returned alone, duty done",
        "knew the sound of the five-thirty bus and objected to it daily",
    ])
    feat = rng.choice([
        "put out a fire by making a noise the fire engine could only envy",
        "found a lost wedding ring in a border collie's time, though slower",
        "led two stranded hikers off the moor in thick fog",
        "stopped a runaway pram at the bottom of Chapel Hill",
        "woke the whole street at four in the morning about a burst water main",
    ])
    honor = rng.choice([
        "received, by popular decree, the freedom of the fish market",
        "was fed thereafter at the town's expense and grew smug about it",
        "got its photograph in the window of the post office, next to the motorbike licenses",
        "was declared, at a meeting nobody minuted, a local institution",
    ])
    return (f"Everyone in {place} knew {animal}, who {habit}.\n\n"
            f"Then came the day {animal} {feat}, and after that nothing was the same: "
            f"{animal} {honor}.\n\n"
            f"Ask anyone in {place} about it today and they will tell you the story in "
            f"roughly the same words, which is how you know the important parts are true. "
            f"Ask {animal}, said the locals, and you will get a look that suggests "
            f"the story has always been more complicated than the humans realize.")


@family("story_journey_errand")
def story_journey_errand(rng: random.Random) -> str:
    name = rng.choice(FIRST_NAMES)
    place = rng.choice(PLACES)
    goal = rng.choice(PLACES)
    while goal == place:
        goal = rng.choice(PLACES)
    cargo = rng.choice(["a jar of preserved plums", "a letter that must not get wet",
                        "a wedding cake in two boxes", "a medicine bottle wrapped in wool",
                        "a violin older than the road", "a fish in a bucket",
                        "eggs, one dozen, entire", "a christening shawl"])
    mode = rng.choice(["by the early bus", "on a borrowed bicycle", "on foot",
                       "by the milk train", "in the back of the post van"])
    weather = rng.choice(WEATHER)
    mishap = rng.choice([
        "the bridge was up and the ferryman was having tea",
        "the road flooded to the depth of one regret",
        "the bus driver forgot the timetable and invented a better one",
        "a flock of sheep entered negotiations with the traffic",
        "the bicycle chain chose the steepest hill to retire on",
    ])
    aid = rng.choice([
        "a truck driver with a flask of tea and no schedule",
        "the schoolmistress driving her mother to market",
        "two choirboys with a rope",
        "the village constable, who insisted this was routine",
        "a honeymoon couple in a decorated car",
    ])
    return (f"When {name} set out from {place} with {cargo}, the plan was simple: "
            f"{mode} to {goal}, deliver, return before dark. It was {weather}, "
            f"but the plan was simple, and simple plans feel weatherproof.\n\n"
            f"Halfway there, {mishap}. Simple plans, it turned out, are only weatherproof "
            f"indoors. It was {aid} who got things moving again, on the understanding that "
            f"{name} would never let the story be forgotten.\n\n"
            f"The {cargo.split(' ')[0]} arrived late, slightly famous. In {goal} they tell "
            f"the story as a warning about travel; in {place} they tell it as proof that "
            f"the world, despite its weather, is mostly full of {aid.split(' with')[0]} "
            f"and other reasonable people.")


DEFAULT_COUNTS = {
    # conversation: warm, everyday (kept modest: conv docs are the longest,
    # so raw counts below still translate into a healthy share)
    "conv_plans": 60, "conv_favor": 55, "conv_help_task": 55, "conv_advice": 55,
    "conv_disagree": 55, "conv_smalltalk": 60, "conv_illness": 50,
    "conv_shopping": 50, "conv_lost_item": 50, "conv_weekend_review": 50,
    "conv_teaching_moment": 50, "conv_phone_invite": 50,
    # qa
    "qa_definition": 130, "qa_why": 120, "qa_how": 120, "qa_diff": 105,
    "qa_what_if": 105, "qa_where": 90, "qa_when": 80, "qa_who": 80,
    "qa_everyday_best": 90,
    # descriptions
    "desc_room": 140, "desc_person": 140, "desc_landscape": 125,
    "desc_weather_scene": 115, "desc_object": 115, "desc_animal": 115,
    "desc_market": 80,
    # instructions
    "instr_recipe": 150, "instr_procedure": 185, "instr_rules": 110,
    "instr_directions": 150,
    # explanations
    "expl_why_long": 190, "expl_how_long": 190, "expl_definition_long": 190,
    "expl_compare": 145, "expl_cause_chain": 130, "expl_math_words": 105,
    # formal
    "formal_letter": 120, "formal_notice": 80, "formal_report": 95,
    # stories (authored narrative; no downloads available in this environment)
    "story_found_object": 180, "story_two_friends": 160, "story_weather_day": 140,
    "story_first_job": 140, "story_small_kindness": 140, "story_animal_day": 120,
    "story_journey_errand": 140,
    # extra story-style variety from mixed knowledge + narrative is provided by seeds
}

FAMILY_CATEGORY = {
    "conv": "conversation", "qa": "questions_and_answers", "desc": "descriptions",
    "instr": "instructions", "expl": "explanations", "formal": "formal",
    "story": "stories",
}


def generate_documents(seed: int = 20260919, counts: Dict[str, int] | None = None):
    """Yield (category, group, text) tuples deterministically."""
    counts = counts or DEFAULT_COUNTS
    from core.dataset import Document
    rng = random.Random(seed)
    for fname in sorted(counts):
        fn = FAMILIES[fname]
        cat = FAMILY_CATEGORY[fname.split("_")[0]]
        for i in range(counts[fname]):
            # each instance gets its own stream derived from seed+family+i
            r = random.Random(f"{seed}:{fname}:{i}")
            text = fn(r).strip()
            yield Document(doc_id=f"gen:{fname}:{i:03d}", category=cat, text=text,
                           group=f"gen:{fname}", origin="generated")
