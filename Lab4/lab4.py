class MusicExpertSystem:
    def __init__(self):
        self.rules = {
            ("happy", "energetic"): ["The Beatles - Abbey Road", "Pharrell Williams - Happy"],
            ("happy", "pop"): ["Michael Jackson - Thriller", "ABBA - Gold"],
            ("sad", "emotional"): ["Have a nice life - Deathcounsciousness", "Death in June - But, What Ends When the Symbols Shatter?"],
            ("sad", "rock"): ["Alice in Chains - Jar of Flies", "Pink Floyd - The Wall"],
            ("relaxed", "acoustic"): ["Jack Johnson - In Between Dreams", "John Mayer - Continuum"],
            ("relaxed", "classical"): ["Mozart - Classical Essentials", "Beethoven - Symphony No. 9"],
            ("energetic", "rock"): ["King Gizzard & The Lizard Wizard - Nonagon Infinity", "Funeral candies - Месу розпочато"],
            ("energetic", "electronic"): ["New order - Blue monday", "The Prodigy - Fat of the Land"],
            ("sad", "electronic"): ["Sewerslvt - Cyberia Lyr1+2=3", "HEALTH - DISCO4", "Oblique occasions - Anathema"],
            ("energetic", "metal"): ["Nine inch nails - The downward spiral", "Bring me the horizon - Count your blessings"],
            ("sad", "metal"): ["Tool - 10000 days", "Deftones - White Pony"],

        }

    def get_recommendations(self):
        print("Welcome to Music Recommendation Expert System!")
        print("\nHow are you feeling? (happy/sad/relaxed/energetic):")
        feeling = input().lower()
        
        print("\nWhat genre do you prefer? (pop/rock/classical/electronic/acoustic/emotional/metal):")
        genre = input().lower()

        recommendation = self.find_matches(feeling, genre)
        self.display_results(recommendation)

    def find_matches(self, feeling, genre):
        exact_matches = []
        partial_matches = []
        
        for rule, albums in self.rules.items():
            # Check if both feeling and genre match the rule
            if feeling in rule and genre in rule:
                exact_matches.extend(albums)
            # Check if either feeling or genre matches the rule
            elif feeling in rule or genre in rule:
                partial_matches.extend(albums)
        
        # Combine exact matches first, then partial matches
        return exact_matches + partial_matches

    def display_results(self, recommendations):
        print("\nBased on your preferences, here are some recommendations:")
        if recommendations:
            for i, album in enumerate(recommendations, 1):
                print(f"{i}. {album}")
        else:
            print("Sorry, no specific recommendations found for your combination.")
            print("Try different feelings or genres!")

# Create and run the expert system
if __name__ == "__main__":
    expert_system = MusicExpertSystem()
    expert_system.get_recommendations()