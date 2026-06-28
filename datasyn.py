import pandas as pd
import numpy as np

def generate_mock_data(n_rows=590):
    np.random.seed(42)
    airlines = ["Delta Air Lines", "Emirates", "Singapore Airlines", "Ryanair", "British Airways", "Lufthansa", "Qatar Airways", "United Airlines", "Zipair", "ANA"]
    aircrafts = ["Boeing 777", "Airbus A380", "Boeing 737", "Airbus A320", "Boeing 787", "Airbus A350", ""]
    travellers = ["Solo Leisure", "Couple Leisure", "Business", "Family Leisure"]
    seats = ["Economy Class", "Business Class", "First Class", "Premium Economy"]
    routes = ["London to New York", "Dubai to Mumbai", "Singapore to Tokyo", "Paris to Berlin", "New York to Los Angeles", "Tokyo to Los Angeles", "Frankfurt to Dubai"]

    review_pool = [
        ("The flight was delayed for 3 hours and the service was bad.", "Terrible delay", "no"),
        ("Great food and very comfortable seat with excellent legroom.", "Amazing flight", "yes"),
        ("The cabin crew and staff were extremely rude. Food was cold.", "Disappointed with staff", "no"),
        ("Entertainment system didn't work, no movie or wifi available.", "Boring flight", "no"),
        ("Lost my baggage at the airport. Very expensive flight for nothing.", "Lost my bags", "no"),
        ("The aircraft was clean and shiny. Excellent value for money.", "Great clean cabin", "yes"),
        ("The seat space was tight but the flight arrived on time.", "Average flight", "yes"),
        ("Flight got cancelled and no snack or drink was provided.", "Cancelled flight horror", "no"),
        ("Legroom was tight, overpriced and late departure.", "Not worth the price", "no"),
        ("Amazing inflight entertainment, watched a great movie. Friendly staff.", "Superb service", "yes"),
        ("The wifi was surprisingly fast, I could work the whole flight.", "Great connectivity", "yes"),
        ("Dirty bathrooms and no hygiene. Completely unacceptable.", "Terrible hygiene", "no"),
        ("Crew was attentive, but the food gave me a stomach ache.", "Mixed experience", "no"),
        ("Plenty of space for my bags, boarding was a breeze.", "Smooth boarding", "yes")
    ]

    data = []
    for i in range(n_rows):
        rev_idx = np.random.choice(len(review_pool))
        review_text, title, recommended = review_pool[rev_idx]
        overall_rating = np.random.randint(7, 11) if recommended == "yes" else np.random.randint(1, 6)
        
        row = {
            "Airline Name": np.random.choice(airlines),
            "Overall_Rating": overall_rating,
            "Review_Title": title,
            "Review Date": f"{np.random.randint(1,29)}th June 2026",
            "Verified": np.random.choice([True, False]),
            "Review": review_text,
            "Aircraft": np.random.choice(aircrafts),
            "Type Of Traveller": np.random.choice(travellers),
            "Seat Type": np.random.choice(seats),
            "Route": np.random.choice(routes),
            "Date Flown": "June 2026",
            "Seat Comfort": np.random.randint(4, 6) if recommended == "yes" else np.random.randint(1, 3),
            "Cabin Staff Service": np.random.randint(4, 6) if recommended == "yes" else np.random.randint(1, 3),
            "Food & Beverages": np.random.randint(3, 6) if recommended == "yes" else np.random.randint(1, 3),
            "Ground Service": np.random.randint(4, 6) if recommended == "yes" else np.random.randint(1, 3),
            "Inflight Entertainment": np.random.randint(3, 6) if recommended == "yes" else np.random.randint(1, 3),
            "Wifi & Connectivity": np.random.randint(3, 6) if recommended == "yes" else np.random.randint(1, 3),
            "Value For Money": np.random.randint(4, 6) if recommended == "yes" else np.random.randint(1, 3),
            "Recommended": recommended
        }
        data.append(row)
    return pd.DataFrame(data)