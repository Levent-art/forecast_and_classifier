import pandas as pd

# Crear los datos
data = {
    "text": [
        # Urgent
        "The server is down and needs to be fixed immediately!",
        "I lost my passport at the airport; please help me now!",
        "There's a fire in the building; evacuate immediately!",
        "This email contains critical updates for the project; read ASAP.",
        "Call the ambulance! There’s been a serious accident on the highway.",
        "The system outage must be resolved before the end of the day.",
        "I need this report finalized within the next hour; it’s urgent.",
        "A critical bug has been reported in production; we need a patch now.",
        "Please reply to my email regarding the payment issue as soon as possible.",
        "This is an emergency; please evacuate the premises immediately!",
        "The deadline has been moved up to tonight; we need to act fast.",
        "My flight is delayed, and I need to reschedule urgently.",
        "There's a security breach in the system that must be resolved now.",
        "I need immediate assistance with this medical issue.",
        "The client has requested an urgent meeting to discuss the proposal.",
        "We are running out of time to complete the project on schedule.",
        "The package must arrive at its destination within the next 24 hours.",
        # Artificial Intelligence
        "Geoffrey Hinton is often called the godfather of deep learning.",
        "GPT models have revolutionized the way machines process human language.",
        "Reinforcement learning has enabled AI to excel at games like Go and chess.",
        "Autonomous cars rely heavily on computer vision and AI algorithms.",
        "Neural networks mimic the structure of the human brain to process data.",
        "OpenAI's DALL-E can generate realistic images from textual descriptions.",
        "Artificial intelligence is transforming industries like healthcare and finance.",
        "AI-based recommendation systems power platforms like Netflix and Amazon.",
        "Machine learning models require large datasets to train effectively.",
        "The ethical implications of AI in surveillance are a growing concern.",
        "ChatGPT has redefined conversational AI with its natural language responses.",
        "AI can predict weather patterns with incredible accuracy using big data.",
        "Natural language processing allows machines to understand human speech.",
        "Deep learning has led to breakthroughs in image recognition and analysis.",
        "AI-powered tools are helping doctors diagnose diseases faster.",
        "Companies are investing heavily in AI research to stay competitive.",
        "The future of AI lies in its ability to learn and adapt autonomously.",
        # Computer
        "My computer crashed, and I lost all my unsaved work!",
        "This laptop features an advanced GPU for gaming and video editing.",
        "The IT department is upgrading the company's desktop computers.",
        "Installing the new software requires at least 16GB of RAM.",
        "I forgot my laptop charger, and now my battery is dead.",
        "Cloud storage solutions allow you to access files from any computer.",
        "My desktop PC is overheating and shutting down randomly.",
        "This external hard drive is compatible with both Mac and Windows computers.",
        "Building a custom PC is popular among gaming enthusiasts.",
        "The motherboard is the central hub that connects all computer components.",
        "The operating system update fixed several bugs and improved performance.",
        "This computer monitor has a refresh rate of 144Hz for smoother gameplay.",
        "My keyboard stopped working, and I can't log into my computer.",
        "The company issued new laptops to all employees for remote work.",
        "This programming course teaches you how to write code for computers.",
        "Graphics cards are essential for rendering high-quality 3D visuals.",
        "My laptop fan is making a loud noise; I think it needs cleaning.",
        # Travel
        "I’m planning a trip to Japan next spring to explore Kyoto and Tokyo.",
        "This travel guide provides tips for visiting Europe on a budget.",
        "Booking a direct flight is often more convenient but expensive.",
        "The hotel offered a free breakfast buffet during our stay.",
        "Traveling to tropical destinations requires packing sunscreen and light clothing.",
        "Backpacking through South America was the adventure of a lifetime.",
        "Paris is often called the city of love and attracts millions of tourists annually.",
        "Exploring the Grand Canyon is a must for any nature enthusiast.",
        "Travel insurance is essential for peace of mind during your trip.",
        "Renting a car allowed us to explore the countryside at our own pace.",
        "I love visiting new cultures and tasting traditional cuisines while traveling.",
        "We missed our connecting flight and had to stay overnight at the airport.",
        "This app helps you find cheap flights and hotel deals for your travels.",
        "Hiking through the Swiss Alps was both challenging and rewarding.",
        "The travel agency planned our itinerary down to the smallest detail.",
        "Venice is famous for its canals and gondola rides.",
        "The tropical beaches of Bali are perfect for a relaxing vacation.",
        # Animal
        "The cheetah is the fastest land animal, capable of speeds up to 70 mph.",
        "Axolotls are unique amphibians that can regenerate their limbs.",
        "Elephants are known for their intelligence and strong family bonds.",
        "Pandas primarily eat bamboo and are considered a vulnerable species.",
        "Owls are nocturnal animals with excellent night vision.",
        "The Siberian tiger is one of the most endangered species in the world.",
        "Dogs are often called man’s best friend due to their loyalty.",
        "Humpback whales migrate thousands of miles every year.",
        "Kangaroos are native to Australia and carry their young in pouches.",
        "Snakes shed their skin multiple times a year as they grow.",
        "Penguins can’t fly but are excellent swimmers.",
        "Cats are known for their agility and independent nature.",
        "Octopuses have three hearts and are highly intelligent creatures.",
        "A lion’s roar can be heard up to 5 miles away in the wild.",
        "Parrots can mimic human speech and are popular as pets.",
        "The giant tortoise can live for over 100 years in the wild.",
        "Bees play a crucial role in pollinating plants and producing honey.",
        # Fiction
        "Harry Potter discovered he was a wizard on his 11th birthday.",
        "In Middle-earth, Frodo Baggins embarks on a journey to destroy the One Ring.",
        "Sherlock Holmes is known for solving mysteries with his keen observation skills.",
        "Dracula is a classic gothic novel about a vampire.",
        "The Hunger Games depicts a dystopian society with a deadly competition.",
        "Alice fell down a rabbit hole into the whimsical world of Wonderland.",
        "The Chronicles of Narnia begins with children discovering a magical wardrobe.",
        "The Great Gatsby explores themes of wealth and the American Dream.",
        "Frankenstein is a tale of a scientist who creates a living monster.",
        "Percy Jackson discovers he is the son of Poseidon in this mythological tale.",
        "The Lord of the Rings is a legendary saga of good versus evil.",
        "Dune is a science fiction masterpiece about political intrigue on a desert planet.",
        "To Kill a Mockingbird addresses issues of race and justice in the American South.",
        "Brave New World imagines a dystopian future with genetic engineering.",
        "The Catcher in the Rye follows the life of a disillusioned teenager.",
        "Game of Thrones is a tale of power struggles in a medieval fantasy world.",
        "Moby-Dick is a story of obsession and revenge between a man and a whale.",
    ],
    "label": ["urgent"] * 17 + ["artificial intelligence"] * 17 + ["computer"] * 17 +
        ["travel"] * 17 + ["animal"] * 17 + ["fiction"] * 17
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Guardar como CSV
file_path = "data/test_data.csv"
df.to_csv(file_path, index=False)

file_path
