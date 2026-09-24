"""
harvest_authors.py — the author list for the targeted Hardcover sweep.

WHY A LIST AND NOT A DEEPER CRAWL
books_v2_hc was paginated by users_count desc to a floor of 13. That returns the
most-read books on a Western reading app, which is why the corpus is 46% fantasy,
30% young adult and 4.1% non-fiction. Going deeper returns more of the same
population. A named list changes WHICH population is sampled, which is the only
thing that moves composition.

The probe measured this: 12 authors, 91 described books in Hardcover and absent
from the corpus. Ram: Scion of Ikshvaku has 7 users; the harvest floor was 13.
Ruskin Bond has 44 books in Hardcover and 1 in the corpus.

SELECTION RULE
One level below BookTok. Widely read, critically serious, and specifically NOT
the titles a popularity sweep already caught. Weighted toward non-fiction and
literary fiction because those are the measured holes -- adding more fantasy
would deepen a skew that is already the problem.

ENGLISH ONLY
Filtering happens at fetch time on the DESCRIPTION, not here: Hardcover holds
one record per book and a translated author's English edition is the one that
carries an English blurb. Mixing registers and languages inside one embedding
space is what put the Russian edition of ReWork next to English business books.

SPELLING MATTERS
The API rejects _ilike with a 403, so lookups use _eq and the name must match
Hardcover's spelling exactly. ALIASES carries the variants worth trying when the
primary spelling returns nothing -- initials with and without spaces are the
usual culprit.
"""

# --- Indian and South Asian literary fiction --------------------------------
INDIAN_LITERARY = [
    "Amitav Ghosh", "Vikram Seth", "Rohinton Mistry", "Anita Desai",
    "Kiran Desai", "Jhumpa Lahiri", "Arundhati Roy", "Salman Rushdie",
    "R.K. Narayan", "Ruskin Bond", "Khushwant Singh", "Shashi Tharoor",
    "Upamanyu Chatterjee", "Amit Chaudhuri", "Jerry Pinto", "Kiran Nagarkar",
    "Perumal Murugan", "Mahasweta Devi", "U.R. Ananthamurthy",
    "Vivek Shanbhag", "Benyamin", "M.T. Vasudevan Nair", "Paul Zacharia",
    "Manto", "Saadat Hasan Manto", "Ismat Chughtai", "Premchand",
    "Bibhutibhushan Bandyopadhyay", "Sarat Chandra Chattopadhyay",
    "Rabindranath Tagore", "Gita Mehta", "Nayantara Sahgal",
    "Githa Hariharan", "Anuradha Roy", "Jeet Thayil", "Aravind Adiga",
    "Manu Joseph", "Cyrus Mistry", "Neel Mukherjee", "Sunjeev Sahota",
    "Mohsin Hamid", "Kamila Shamsie", "Mohammed Hanif", "Nadeem Aslam",
    "Bapsi Sidhwa", "Michael Ondaatje", "Romesh Gunesekera",
    "Shehan Karunatilaka", "Tahmima Anam", "Monica Ali",
]

# --- Indian genre, mythology, commercial ------------------------------------
INDIAN_POPULAR = [
    "Amish Tripathi", "Devdutt Pattanaik", "Chetan Bhagat", "Ashwin Sanghi",
    "Anand Neelakantan", "Kevin Missal", "Krishna Udayasankar",
    "S. Hussain Zaidi", "Hussain Zaidi", "Vish Dhamija", "Ravi Subramanian",
    "Anuja Chauhan", "Twinkle Khanna", "Durjoy Datta", "Preeti Shenoy",
]

# --- Narrative non-fiction and journalism (the 4.1% hole) -------------------
NONFICTION_NARRATIVE = [
    "Katherine Boo", "Sonia Faleiro", "Rana Dasgupta", "Suketu Mehta",
    "Siddhartha Mukherjee", "Abraham Verghese", "Atul Gawande",
    "Pankaj Mishra", "Ramachandra Guha", "Sunil Khilnani", "Patrick French",
    "William Dalrymple", "Mukul Kesavan", "Josy Joseph", "Rahul Pandita",
    "Barkha Dutt", "Raghu Karnad", "Aanchal Malhotra", "Ornit Shani",
    "Erik Larson", "Jon Krakauer", "Susan Orlean", "David Grann",
    "Patrick Radden Keefe", "Michael Lewis", "Robert Caro", "Barbara Demick",
    "Ed Yong", "Mary Roach", "Rebecca Skloot", "Bill Bryson",
    "Anthony Bourdain", "Michael Pollan", "Helen Macdonald",
    "Robert Macfarlane", "Rebecca Solnit", "Olivia Laing", "Maggie Nelson",
    "Svetlana Alexievich", "Peter Hessler", "Adam Higginbotham",
]

# --- Literary fiction in translation (English editions) ---------------------
TRANSLATED = [
    "Roberto Bolaño", "Javier Marías", "Elena Ferrante", "Italo Calvino",
    "Primo Levi", "Natalia Ginzburg", "Cesare Pavese", "Clarice Lispector",
    "Machado de Assis", "Jorge Amado", "Juan Rulfo", "Mario Vargas Llosa",
    "Gabriel García Márquez", "Julio Cortázar", "César Aira", "Samanta Schweblin",
    "Mariana Enriquez", "Fernanda Melchor", "Valeria Luiselli",
    "Han Kang", "Bora Chung", "Cho Nam-Joo", "Hwang Sok-yong",
    "Yoko Ogawa", "Sayaka Murata", "Mieko Kawakami", "Hiromi Kawakami",
    "Kobo Abe", "Yasunari Kawabata", "Yukio Mishima", "Natsume Sōseki",
    "Can Xue", "Yu Hua", "Yan Lianke", "Mo Yan", "Eileen Chang",
    "Olga Tokarczuk", "László Krasznahorkai", "Magda Szabó", "Bruno Schulz",
    "Andrey Kurkov", "Mikhail Bulgakov", "Vasily Grossman",
    "Orhan Pamuk", "Elif Shafak", "Hanan al-Shaykh", "Ghassan Kanafani",
    "Tayeb Salih", "Naguib Mahfouz", "Alaa Al Aswany",
    "Ngũgĩ wa Thiong'o", "Chinua Achebe", "Ben Okri", "Tsitsi Dangarembga",
    "Maaza Mengiste", "Abdulrazak Gurnah", "Leila Slimani", "Annie Ernaux",
    "Patrick Modiano", "Georges Perec", "Marguerite Duras", "Jean-Patrick Manchette",
    "W.G. Sebald", "Thomas Bernhard", "Peter Handke", "Herta Müller",
    "Jenny Erpenbeck", "Judith Schalansky", "Karl Ove Knausgård",
    "Tove Jansson", "Tove Ditlevsen", "Halldór Laxness", "Jon Fosse",
]

# --- Canadian and Indigenous ------------------------------------------------
CANADIAN_INDIGENOUS = [
    "Thomas King", "Lee Maracle", "Eden Robinson", "Richard Wagamese",
    "Tanya Talaga", "Tomson Highway", "Katherena Vermette", "Cherie Dimaline",
    "Waubgeshig Rice", "Michelle Good", "Joshua Whitehead", "Billy-Ray Belcourt",
    "Tommy Orange", "Louise Erdrich", "Stephen Graham Jones", "N. Scott Momaday",
    "Leslie Marmon Silko", "Robin Wall Kimmerer",
    "Alice Munro", "Margaret Atwood", "Mordecai Richler", "Robertson Davies",
    "Miriam Toews", "Sheila Heti", "Rawi Hage", "Esi Edugyan",
    "Madeleine Thien", "David Chariandy", "Souvankham Thammavongsa",
]

# --- Literary crime and noir ------------------------------------------------
CRIME_LITERARY = [
    "Dashiell Hammett", "Raymond Chandler", "James M. Cain", "Ross Macdonald",
    "Patricia Highsmith", "Jim Thompson", "Chester Himes", "Elmore Leonard",
    "George V. Higgins", "Richard Price", "Dennis Lehane", "James Ellroy",
    "Megan Abbott", "Tana French", "Kate Atkinson", "Denise Mina",
    "Fred Vargas", "Andrea Camilleri", "Jo Nesbø", "Henning Mankell",
    "Keigo Higashino", "Seicho Matsumoto", "Natsuo Kirino",
]

# --- Literary speculative, horror, SF one level below the bestseller list ----
SPECULATIVE_LITERARY = [
    "Ursula K. Le Guin", "Octavia E. Butler", "Samuel R. Delany",
    "Gene Wolfe", "M. John Harrison", "China Miéville", "Jeff VanderMeer",
    "Kelly Link", "Carmen Maria Machado", "Shirley Jackson", "Angela Carter",
    "Brian Evenson", "Thomas Ligotti", "Robert Aickman", "Laird Barron",
    "Ted Chiang", "Greg Egan", "Stanisław Lem", "Arkady Strugatsky",
    "Liu Cixin", "Hao Jingfang", "Chen Qiufan",
    "Nnedi Okorafor", "Tade Thompson", "Namwali Serpell", "Rivers Solomon",
    "Indra Das", "Vandana Singh", "Samit Basu", "Prayaag Akbar",
]

# --- Essays, criticism, memoir ----------------------------------------------
ESSAY_MEMOIR = [
    "Joan Didion", "James Baldwin", "Zadie Smith", "Teju Cole",
    "Hilton Als", "Vivian Gornick", "Janet Malcolm", "Ta-Nehisi Coates",
    "Jia Tolentino", "Leslie Jamison", "Eula Biss", "John McPhee",
    "Geoff Dyer", "Ian Frazier", "Pico Iyer", "Amitava Kumar",
    "Deborah Levy", "Annie Dillard", "Terry Tempest Williams",
]

ALL_AUTHORS = (
    INDIAN_LITERARY + INDIAN_POPULAR + NONFICTION_NARRATIVE + TRANSLATED
    + CANADIAN_INDIGENOUS + CRIME_LITERARY + SPECULATIVE_LITERARY + ESSAY_MEMOIR
)

CATEGORIES = {
    "indian_literary": INDIAN_LITERARY,
    "indian_popular": INDIAN_POPULAR,
    "nonfiction_narrative": NONFICTION_NARRATIVE,
    "translated": TRANSLATED,
    "canadian_indigenous": CANADIAN_INDIGENOUS,
    "crime_literary": CRIME_LITERARY,
    "speculative_literary": SPECULATIVE_LITERARY,
    "essay_memoir": ESSAY_MEMOIR,
}

# Tried in order when the primary spelling returns nothing. _eq is exact, so a
# space between initials is the difference between 32 books and zero.
ALIASES = {
    "R.K. Narayan": ["R. K. Narayan", "RK Narayan"],
    "U.R. Ananthamurthy": ["U. R. Ananthamurthy", "U R Ananthamurthy"],
    "M.T. Vasudevan Nair": ["M. T. Vasudevan Nair"],
    "W.G. Sebald": ["W. G. Sebald"],
    "N. Scott Momaday": ["N Scott Momaday"],
    "Manto": ["Saadat Hasan Manto"],
    "Hussain Zaidi": ["S. Hussain Zaidi", "S Hussain Zaidi"],
    "Ngũgĩ wa Thiong'o": ["Ngugi wa Thiong'o", "Ngũgĩ wa Thiongo"],
    "Roberto Bolaño": ["Roberto Bolano"],
    "Javier Marías": ["Javier Marias"],
    "Karl Ove Knausgård": ["Karl Ove Knausgaard"],
    "Jo Nesbø": ["Jo Nesbo"],
    "Stanisław Lem": ["Stanislaw Lem"],
    "Natsume Sōseki": ["Natsume Soseki"],
    "Halldór Laxness": ["Halldor Laxness"],
    "Magda Szabó": ["Magda Szabo"],
    "László Krasznahorkai": ["Laszlo Krasznahorkai"],
    "Gabriel García Márquez": ["Gabriel Garcia Marquez"],
    "Mariana Enriquez": ["Mariana Enríquez"],
    "Cho Nam-Joo": ["Cho Nam-joo", "Nam-Joo Cho"],
    "Hwang Sok-yong": ["Hwang Sok-Yong"],
    "Liu Cixin": ["Cixin Liu"],
}


if __name__ == "__main__":
    print(f"{len(ALL_AUTHORS)} authors, {len(set(ALL_AUTHORS))} unique\n")
    for name, lst in CATEGORIES.items():
        print(f"  {name:<24} {len(lst):>4}")
    dupes = {a for a in ALL_AUTHORS if ALL_AUTHORS.count(a) > 1}
    if dupes:
        print(f"\n  duplicates across categories: {sorted(dupes)}")
