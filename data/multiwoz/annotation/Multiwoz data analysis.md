## Multiwoz data analysis

source: https://www.repository.cam.ac.uk/handle/1810/280608

> ##### Description
>
> Dataset contains the following json files: 1. data.json: the woz dialogue dataset, which contains the conversation users and wizards, as well as a set of coarse labels for each user turn. 2. restaurant_db.json: the Cambridge restaurant database file, containing restaurants in the Cambridge UK area and a set of attributes. 3. attraction_db.json: the Cambridge attraction database file, contining attractions in the Cambridge UK area and a set of attributes. 4. hotel_db.json: the Cambridge hotel database file, containing hotels in the Cambridge UK area and a set of attributes. 5. train_db.json: the Cambridge train (with artificial connections) database file, containing trains in the Cambridge UK area and a set of attributes. 6. hospital_db.json: the Cambridge hospital database file, contatining information about departments. 7. police_db.json: the Cambridge police station information. 8. taxi_db.json: slot-value list for taxi domain. 9. valListFile.json: list of dialogues for validation. 10. testListFile.json: list of dialogues for testing. 11. system_acts.json: system acts annotations 12. ontology.json: Data-based ontology.
>
> ##### Format
>
> The Multi-Domain Wizard-of-Oz dataset (MultiWOZ), a collection of human-human written conversations spanning over multiple domains and topics. The dataset was collected based on the Wizard of Oz experiment on Amazon MTurk. Each dialogue contains a goal label and several exchanges between a visitor and the system. Each system turn has labels from the set of slot-value pairs representing a coarse representation of dialogue state for both user and system. There are in total 10438 dialogues.

### Database

Contain 110 restaurants, 79 attractions, 33 hotels, 2828 trains, 66 hospitals, 1 police and generator of taxi.

#### Restaurant

110 restaurants, example:

```json
{
    "address": "Finders Corner Newmarket Road",
    "area": "east",
    "food": "international",
    "id": "30650",
    "introduction": "",
    "location": [
        52.21768,
        0.224907
    ],
    "name": "the missing sock",
    "phone": "01223812660",
    "postcode": "cb259aq",
    "pricerange": "cheap",
    "signature": "african babooti",
    "type": "restaurant"
}
```
There are 9 attributes that all restaurants have: 'address', 'area', 'food', 'id', 'location', 'name', 'postcode', 'pricerange', 'type'. 3 attributes may be missing: 'introduction', 'phone', 'signature'.

#### Attraction

79 attractions, example:

```json
{
    "address": "pool way, whitehill road, off newmarket road",
    "area": "east",
    "entrance fee": "?",
    "id": "1",
    "location": [
        52.208789,
        0.154883
    ],
    "name": "abbey pool and astroturf pitch",
    "openhours": "?",
    "phone": "01223902088",
    "postcode": "cb58nt",
    "pricerange": "?",
    "type": "swimmingpool"
}
```
#### Hotel

33 hotels, example:

```json
{
    "address": "124 tenison road",
    "area": "east",
    "internet": "yes",
    "parking": "no",
    "id": "0",
    "location": [
        52.1963733,
        0.1987426
    ],
    "name": "a and b guest house",
    "phone": "01223315702",
    "postcode": "cb12dp",
    "price": {
        "double": "70",
        "family": "90",
        "single": "50"
    },
    "pricerange": "moderate",
    "stars": "4",
    "takesbookings": "yes",
    "type": "guesthouse"
}
```
#### Train

2828 train (with artificial connections), example:

```json
{
    "arriveBy": "05:51",
    "day": "monday",
    "departure": "cambridge",
    "destination": "london kings cross",
    "duration": "51 minutes",
    "leaveAt": "05:00",
    "price": "23.60 pounds",
    "trainID": "TR7075"
}
```
#### Hospital

66 hospitals, example:

```json
{
    "department": "neurosciences critical care unit",
    "id": 0,
    "phone": "01223216297"
}
```
#### Police

1 police:

```json
{
    "name": "Parkside Police Station",
    "address": "Parkside, Cambridge",
    "id": 0,
    "phone": "01223358966"
}
```
#### Taxi

slot-value list for taxi domain.

```json
[
  "taxi_colors" : ["black","white","red","yellow","blue",'grey'],
  "taxi_types":  ["toyota","skoda","bmw",'honda','ford','audi','lexus','volvo','volkswagen','tesla']
  "taxi_phone": ["^[0-9]{10}$"]
]
```

### Ontology

35 slots: (not reliable, lack some slots appeared in dialog_acts.json, maybe only have the slots that used in goal generation)

```
'hotel-price range', 
'hotel-type', 
'hotel-parking', 
'hotel-book stay', 
'hotel-book day', 
'hotel-book people', 
'hotel-area', 
'hotel-stars', 
'hotel-internet', 
'hotel-name', 
=====================================
'train-destination', 
'train-day', 
'train-departure', 
'train-arrive by', 
'train-book people', 
'train-leave at', 
=====================================
'restaurant-food', 
'restaurant-price range', 
'restaurant-area',
'restaurant-name', 
'restaurant-book time', 
'restaurant-book day', 
'restaurant-book people', 
=====================================
'attraction-area', 
'attraction-name', 
'attraction-type', 
=====================================
'taxi-leave at', 
'taxi-destination', 
'taxi-departure', 
'taxi-arrive by', 
=====================================
'hospital-department', 
=====================================
'bus-departure', 
'bus-destination', 
'bus-leaveAt', 
'bus-day'
```

### Dialog Act

6+1 domains: Booking, Restaurant, Hotel, Attraction, Taxi, Train and general.

Bus, Hospital, Police don't have dialog act. 

```python
Counter({'Attraction-Inform': 6976,
         'Attraction-NoOffer': 490,
         'Attraction-Recommend': 1451,
         'Attraction-Request': 1641,
         'Attraction-Select': 438,
         'Booking-Book': 5256,
         'Booking-Inform': 5703,
         'Booking-NoBook': 1313,
         'Booking-Request': 2708,
         'Hotel-Inform': 8224,
         'Hotel-NoOffer': 914,
         'Hotel-Recommend': 1501,
         'Hotel-Request': 3215,
         'Hotel-Select': 1005,
         'No Annotation': 1933,
         'Restaurant-Inform': 8071,
         'Restaurant-NoOffer': 1453,
         'Restaurant-Recommend': 1495,
         'Restaurant-Request': 3083,
         'Restaurant-Select': 918,
         'Taxi-Inform': 2087,
         'Taxi-Request': 1613,
         'Train-Inform': 7204,
         'Train-NoOffer': 117,
         'Train-OfferBook': 3032,
         'Train-OfferBooked': 2309,
         'Train-Request': 5522,
         'Train-Select': 389,
         'general-bye': 9107,
         'general-greet': 2021,
         'general-reqmore': 13773,
         'general-welcome': 4786})
```

**ATTENTION**: It only contains dialog act and slot from system (or wizard) side. Slot from user such as "Attraction-Request-Post" are not listed! We must design by ourselves.

Update: see `dialog_act_slot.txt` or `Multiwoz_analysis.ipynb` for dialog act with slot.

### User Goal

Example:

```json
"goal": {
"taxi": {
    "info": {
        "arriveBy": "13:15"
    }, 
    "reqt": [
        "car type", 
        "phone"
    ], 
    "fail_info": {}
}, 
"police": {}, 
"hospital": {}, 
"hotel": {}, 
"topic": {
    "taxi": false, 
    "police": false, 
    "restaurant": false, 
    "hospital": false, 
    "hotel": false, 
    "general": false, 
    "attraction": false, 
    "train": false, 
    "booking": false
}, 
"attraction": {
    "info": {
        "name": "the place"
    }, 
    "reqt": [
        "postcode"
    ], 
    "fail_info": {}
}, 
"train": {}, 
"message": [
    "You are looking for information in Cambridge", 
    "You are looking for a **particular attraction**. Its name is called **the place**", 
    "Make sure you get **postcode**", 
    "You are also looking for a **place to dine**. The restaurant should be in the **centre** and should serve **modern european** food", 
    "The restaurant should be in the **moderate** price range", 
    "Once you find the **restaurant** you want to book a table for **4 people** at **13:15** on **saturday**", 
    "Make sure you get the **reference number**", 
    "You also want to book a **taxi** to commute between the two places", 
    "You want to make sure it arrives the restaurant **by the booked time**", 
    "Make sure you get **contact number** and **car type**"
], 
"restaurant": {
    "info": {
        "food": "modern european", 
        "pricerange": "moderate", 
        "area": "centre"
    }, 
    "fail_info": {}, 
    "book": {
        "people": "4", 
        "day": "saturday", 
        "invalid": false, 
        "time": "13:15"
    }, 
    "fail_book": {}
}
}, 
```

Not all slots for a domain have value, when system ask, user can just reply "don't care".

All attributes occured in goal:

```python
attraction
{'info': {'area', 'name', 'type'}, 'reqt': {'entrance fee', 'address', 'phone', 'area', 'postcode', 'type'}, 'fail_info': {'area', 'name', 'type'}}

hospital
{'info': {'department'}, 'reqt': {'address', 'phone', 'postcode'}, 'fail_info': set()}

hotel
{'info': {'name', 'parking', 'internet', 'area', 'stars', 'pricerange', 'type'}, 'fail_info': {'name', 'parking', 'internet', 'area', 'stars', 'pricerange', 'type'}, 'book': {'day', 'pre_invalid', 'people', 'stay', 'invalid'}, 'fail_book': {'day', 'stay'}, 'reqt': {'phone', 'parking', 'internet', 'address', 'area', 'stars', 'postcode', 'pricerange', 'type'}}

police
{'info': set(), 'reqt': {'address', 'phone', 'postcode'}, 'fail_info': set()}

restaurant
{'info': {'food', 'area', 'name', 'pricerange'}, 'reqt': {'phone', 'address', 'food', 'area', 'postcode', 'pricerange'}, 'fail_info': {'area', 'food', 'name', 'pricerange'}, 'book': {'day', 'pre_invalid', 'people', 'time', 'invalid'}, 'fail_book': {'day', 'time'}}

taxi
{'info': {'departure', 'leaveAt', 'destination', 'arriveBy'}, 'reqt': {'phone', 'car type'}, 'fail_info': set()}

train
{'info': {'departure', 'day', 'leaveAt', 'destination', 'arriveBy'}, 'reqt': {'leaveAt', 'price', 'arriveBy', 'trainID', 'duration'}, 'fail_info': set(), 'book': {'people', 'invalid'}, 'fail_book': set()}
```

### Dialog state

**The information provided by user (given in user goal) defines in 'semi' dict for each domain. The information that user need is not showed in dialog state.** 

Full state example:

```json
"text": "Enjoy your stay at the hamilton lodge. Have a great day.", 
"metadata": {
    "taxi": {
        "book": {
            "booked": [
                {
                    "phone": "07218068540", 
                    "type": "blue honda"
                }
            ]
        }, 
        "semi": {
            "leaveAt": "17:15", 
            "destination": "pizza hut fen ditton", 
            "departure": "saint john's college", 
            "arriveBy": "not mentioned"
        }
    }, 
    "police": {
        "book": {
            "booked": []
        }, 
        "semi": {}
    }, 
    "restaurant": {
        "book": {
            "booked": [
                {
                    "name": "pizza hut city centre", 
                    "reference": "F3K2PQZZ"
                }
            ], 
            "time": "19:45", 
            "day": "thursday", 
            "people": "2"
        }, 
        "semi": {
            "food": "not mentioned", 
            "pricerange": "not mentioned", 
            "name": "pizza hut city centre", 
            "area": "not mentioned"
        }
    }, 
    "hospital": {
        "book": {
            "booked": [
                {
                    "department": "infusion services", 
                    "reference": "L2127Y9Q", 
                    "time": "14:00, next Thursday"
                }
            ]
        }, 
        "semi": {
            "department": "infusion services"
        }
    }, 
    "hotel": {
        "book": {
            "booked": [
                {
                    "name": "hamilton lodge", 
                    "reference": "TGSVA9OC"
                }
            ], 
            "people": "1", 
            "day": "sunday", 
            "stay": "5"
        }, 
        "semi": {
            "name": "not mentioned", 
            "area": "not mentioned", 
            "parking": "yes", 
            "pricerange": "moderate", 
            "stars": "3", 
            "internet": "not mentioned", 
            "type": "guesthouse"
        }
    }, 
    "attraction": {
        "book": {
            "booked": []
        }, 
        "semi": {
            "type": "museum", 
            "name": "not mentioned", 
            "area": "east"
        }
    }, 
    "train": {
        "book": {
            "booked": [
                {
                    "trainID": "TR1567", 
                    "reference": "UIFV8FAS"
                }
            ], 
            "people": "1"
        }, 
        "semi": {
            "leaveAt": "not mentioned", 
            "destination": "bishops stortford", 
            "day": "friday", 
            "arriveBy": "19:45", 
            "departure": "cambridge"
        }
    }
}
```

### Automaic Annotation for User Dialog Act

Compare with dialog acts for system:

|                           | System (come from data)                                      | User (defined according to user goal and dialog state table) |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| Attraction-Inform         | Area Type Choice Post Name Phone Addr Fee Open Price none | Area Type Name none |
| Hotel-Inform              | Name Ref Type Choice Addr Post Area Internet Parking Stars Phone Price none | Name Type Area Internet Parking Stars Price Day People Stay none |
| Restaurant-Inform         | Post Food Name Price Addr Phone Area Choice Ref none | Food Name Price Area Time Day People none    |
| Taxi-Inform               | Phone Car Depart Arrive Dest Leave none                            | Depart Dest Leave Arrive none                         |
| Train-Inform              | Arrive Id Leave Time Dest Ticket Depart Day Choice Ref People none      | Depart Dest Leave Arrive Day People none |
| Hospital-Inform | lack annotation | Department none                                    |
| Police-Inform             | lack annotation                                              | none                                              |
| Booking-Inform            | Time Stay Name Day People Ref none                                 | don't need                  |
| Attraction-Request        | Area Type Price Name                                            | Area Type Post Phone Addr Fee                          |
| Hotel-Request             | Area Price Stars Type Parking Internet Name                        | Ref Type Addr Post Area Internet Parking Stars Phone Price Ref |
| Restaurant-Request        | Area Name Price Food                                            | Post Food Price Addr Phone Area Ref                     |
| Taxi-Request              | Depart Arrive Dest Leave                                        | Car Phone                                          |
| Train-Request             | Leave Day Depart Dest Arrive People                               | Id Arrive Leave Time Dest Ticket Depart Ref              |
| Hospital-Request          | lack annotation                                              | Post Addr Phone Ref                                 |
| Police-Request            | lack annotation                                              | Post Addr Phone                                     |
| Booking-Request           | Time Day People Stay                                            | don't need                                      |
| Train-Offer[Book\|Booked] |                                                              | don't need                                        |
| Booking-[No]Book          |                                                              | don't need                                        |
| *-NoOffer                 |                                                              | don't need                                        |
| *-Recommend               |                                                              | don't need                                        |
| *-Select                  |                                                              | don't need                                        |
| general-*                 |                                                              | by keywords?                                      |

User dialog acts include:

- **Inform**: annotated by dialog state update.
- **Request**: implied by system inform, not accurate.
- **General**: keywords detection.

For clarity, some details are not showed below, please refer to the code and comments.

#### Inform

When dialog state changes, the information must come from user by **Inform**. That means when user informs something (from user goal) that in the dialog state table, we can always annotate it. However, there are some problems:

- Can't annotate `Request`, `General` dialog act.
- There are few wrong annotations caused by wrong dialog state update. 

Information comes from dialog state, user goal, utterance, system dialog act.

1. Difference of dialog state.
   1. Value is mentioned by user.
   2. Slot is mentioned by user.
   3. Answer system request ("Would you prefer British, Indian, or Chinese food, or a gastropub?"-"I don't have a preference.").
   4. Value for that slot in the goal is mentioned by user, which indicate wrong state update.
2. state don't change but value is in the goal.
   1. Binary value ("yes" or "no"), if '?' not in user utterance (not for confirmation "Does it have free wifi?").
   2. Digital value (for people, stars...), if value and slot are mentioned in user utterance and domain is mentioned in user utterance or lastest system utterance.
   3. String value, if value are mentioned in user utterance and domain is mentioned in user utterance or lastest system utterance.



#### Request

The value that user request will not be in the dialog state table. We may use user goal, user utterance, system response and its dialog act. Usually, user request includes but not limited to user goal.

1. Slot for request that in user goal and appear in user utterance. For the slot that multiple domain have (area, address...), reserve the one that informed or recommended by system in next utterance.
2. Slot "Ref" if " reference" or "ref" appear in user utterance



#### General

First, if any domain mentioned in user utterance, the dialog act should be `DOMAIN-Inform:[["none", "none"]]`. Then, if there is no other dialog act mentioned above, consider the following:

- Bye: 'bye' in user utterance. Many of them have "thank". Example: "Thanks so much for your help.  Bye."
- Thank:  'bye' not in user utterance but 'thank' in user utterance. Example: "Thank you. I also need a taxi two commute between the hotel and restaurant." (This should be Inform)
- Greet: 'hello', 'hi' in user utterance. Example: "Hello, I'd like to get some info one a hotel please."



#### Result



#### Tokenization

Use [spaCy](https://spacy.io/) for tokenization.

```python
import spacy
nlp = spacy.load('en')
doc = nlp(turn['text'])
turn['text'] = ' '.join([token.text for token in doc]).strip()
```



#### Span info

For slots of **Inform, Select, Recommend, Book, OfferBook, NoBook, NoOffer** dialog act that have non-binary value, use rule to annotate the value occurrence. Cover 98.29% of these slots.

- Exactly match
- Digit alias ("one" for 1)
- Coreference-"same" ("same area")
- Don't care ("doesn't matter")
- Time ("after 2:30")
- Alias "center" for "centre"







#### Police and Hospital

Add annotation for police and hospital domain using the corresponding database.