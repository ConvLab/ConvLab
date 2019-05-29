# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import re
from difflib import SequenceMatcher

init_belief_state = {
        "police": {
            "book": {
                "booked": []
            },
            "semi": {}
        },
        "hotel": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "stay": ""
            },
            "semi": {
                "name": "",
                "area": "",
                "parking": "",
                "pricerange": "",
                "stars": "",
                "internet": "",
                "type": ""
            }
        },
        "attraction": {
            "book": {
                "booked": []
            },
            "semi": {
                "type": "",
                "name": "",
                "area": "",
                "entrance fee": ""
            }
        },
        "restaurant": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "time": ""
            },
            "semi": {
                "food": "",
                "pricerange": "",
                "name": "",
                "area": "",
            }
        },
        "hospital": {
            "book": {
                "booked": []
            },
            "semi": {
                "department": ""
            }
        },
        "taxi": {
            "book": {
                "booked": [],
                "departure": "",
                "destination": ""
            },
            "semi": {
                "leaveAt": "",
                "arriveBy": ""
            }
        },
        "train": {
            "book": {
                "booked": [],
                "people": ""
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "day": "",
                "arriveBy": "",
                "departure": ""
            }
        }
    }


def init_state():
    """
    The init state to start a session.
    Example:
    state = {
            'user_action': None,
            'history': [],
            'belief_state': None,
            'request_state': {}
        }
    """
    # user_action = {'general-hello':{}}
    user_action = {}
    state = {'user_action': user_action,
             'belief_state': init_belief_state,
             'request_state': {},
             'history': []}
    return state

def str_similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normalize_value(value_set, domain, slot, value):
    """
    Normalized the value produced by NLU module to map it to the ontology value space.
    Args:
        value_set (dict): The value set of task ontology.
        domain (str): The domain of the slot-value pairs.
        slot (str): The slot of the value.
        value (str): The raw value detected by NLU module.

    Returns:
        value (str): The normalized value, which fits with the domain ontology.
    """
    slot = slot.lower()
    value = value.lower()
    try:
        assert domain in value_set
    except:
        raise Exception('domain <{}> not found in value set'.format(domain))
    if slot not in value_set[domain]:
        print(value_set[domain].keys())
        raise Exception('slot <{}> not found in db_values[{}]'.format(slot, domain))
    value_list = value_set[domain][slot]
    # for time type slots
    if slot in ['leaveat', 'arriveby']:
        mat = re.search(r"(\d{1,2}:\d{1,2})", value)
        if mat is not None:
            value = mat.groups()[0]
        else:
            value = "00:00" # TODO: check default value
        return value
    # for entrance fee
    if slot == 'entrance fee':
        if 'free' in value:
            return 'free'
        mat = re.search(r"(\d{1}.\d{1,2}) pounds", value)
        if mat is not None:
            value = mat.groups()[0]
            return value
        mat = re.search(r"(\d{1}) pounds", value)
        if mat is not None:
            value = mat.groups()[0]
            return value
        return '5 pounds'  # TODO: check deafult value
    # for ideal condition
    elif value in value_list:
        return value
    # for fuzzy value recognition
    else:
        best_value = value
        best_score = -1
        for v1 in value_list:
            score = str_similar(value, v1)
            if score > best_score:
                best_score = score
                best_value = v1
        with open('fuzzy_recognition.log', 'a+') as f:
            f.write('{} -> {}\n'.format(value, best_value))
        return best_value

if __name__ == "__main__":
    value_set = json.load(open('../../../data/multiwoz/db/db_values.json'))
    print(normalize_value(value_set, 'restaurant', 'address', 'regent street city center'))