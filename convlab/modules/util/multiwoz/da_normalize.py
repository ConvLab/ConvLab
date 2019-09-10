from copy import deepcopy

valid_intent2slot = {}

valid_intent2slot['usr'] = {
    'Attraction-Inform': {'Name', 'Type', 'Area', 'none'},
    'Attraction-Request': {'Fee', 'Post', 'Addr', 'Area', 'Type', 'Phone'},
    'Hospital-Inform': {'Post', 'Phone', 'Department', 'none'},
    'Hospital-Request': {'Post', 'Phone', 'Addr'},
    'Hotel-Inform': {'Stay', 'Day', 'Internet', 'Area', 'Name', 'Parking', 'Type', 'Stars', 'none', 'People', 'Price'},
    'Hotel-Request': {'Post', 'Addr', 'Internet', 'Price', 'Ref', 'Area', 'Type', 'Parking', 'Stars', 'Phone'},
    'Police-Inform': {'Phone', 'Name', 'none', 'Addr'},
    'Police-Request': {'Post', 'Phone', 'Addr'},
    'Restaurant-Inform': {'Food', 'Time', 'Day', 'Area', 'Name', 'none', 'People', 'Price'},
    'Restaurant-Request': {'Food', 'Post', 'Addr', 'Price', 'Ref', 'Area', 'Phone'},
    'Taxi-Inform': {'Leave', 'Dest', 'none', 'Arrive', 'Depart'},
    'Taxi-Request': {'Car', 'Phone'},
    'Train-Inform': {'Leave', 'Day', 'Dest', 'none', 'People', 'Arrive', 'Depart'},
    'Train-Request': {'Time', 'Leave', 'Ref', 'Ticket', 'Id', 'Arrive'},
    'general-bye': {'none'},
    'general-greet': {'none'},
    'general-thank': {'none'}
}

valid_intent2slot['sys'] = {
    'Attraction-Inform': {'Fee', 'Post', 'Addr', 'Open', 'Price', 'Area', 'Name', 'Type', 'none', 'Choice', 'Phone'},
    'Attraction-NoOffer': {'Addr', 'Area', 'Name', 'Type', 'none', 'Choice'},
    'Attraction-Recommend': {'Fee', 'Post', 'Addr', 'Open', 'Price', 'Area', 'Name', 'Type', 'none', 'Choice', 'Phone'},
    'Attraction-Request': {'Name', 'Type', 'Price', 'Area'},
    'Attraction-Select': {'Fee', 'Addr', 'Price', 'Area', 'Name', 'Type', 'none', 'Choice', 'Phone'},
    'Booking-Book': {'Time', 'Stay', 'Day', 'Ref', 'Name', 'none', 'People'},
    'Booking-Inform': {'Time', 'Stay', 'Day', 'Ref', 'Name', 'none', 'People'},
    'Booking-NoBook': {'Time', 'Stay', 'Day', 'Ref', 'Name', 'none', 'People'},
    'Booking-Request': {'Day', 'Time', 'Stay', 'People'},
    'Hospital-Inform': {'Post', 'Phone', 'Department', 'Addr'},
    'Hospital-Request': {'Department'},
    'Hotel-Inform': {'Post', 'Addr', 'Internet', 'Price', 'Ref', 'Area', 'Name', 'Parking', 'Type', 'Stars', 'none', 'Choice', 'Phone'},
    'Hotel-NoOffer': {'Internet', 'Area', 'Name', 'Parking', 'Type', 'Stars', 'none', 'Choice', 'Price'},
    'Hotel-Recommend': {'Post', 'Addr', 'Internet', 'Price', 'Area', 'Name', 'Parking', 'Type', 'Stars', 'none', 'Choice', 'Phone'},
    'Hotel-Request': {'Internet', 'Area', 'Name', 'Parking', 'Type', 'Stars', 'Price'},
    'Hotel-Select': {'Addr', 'Internet', 'Price', 'Area', 'Name', 'Parking', 'Type', 'Stars', 'none', 'Choice', 'Phone'},
    'Police-Inform': {'Phone', 'Name', 'Post', 'Addr'},
    'Restaurant-Inform': {'Food', 'Post', 'Addr', 'Price', 'Ref', 'Area', 'Name', 'none', 'Choice', 'Phone'},
    'Restaurant-NoOffer': {'Food', 'Area', 'Name', 'none', 'Choice', 'Price'},
    'Restaurant-Recommend': {'Food', 'Post', 'Addr', 'Price', 'Area', 'Name', 'none', 'Choice', 'Phone'},
    'Restaurant-Request': {'Food', 'Name', 'Price', 'Area'},
    'Restaurant-Select': {'Food', 'Addr', 'Area', 'Name', 'none', 'Choice', 'Price'},
    'Taxi-Inform': {'Leave', 'Car', 'Dest', 'Phone', 'none', 'Arrive', 'Depart'},
    'Taxi-Request': {'Dest', 'Arrive', 'Depart', 'Leave'},
    'Train-Inform': {'Time', 'Leave', 'Day', 'Dest', 'Ref', 'Ticket', 'none', 'Choice', 'People', 'Id', 'Arrive', 'Depart'},
    'Train-NoOffer': {'Leave', 'Day', 'Dest', 'none', 'Choice', 'Id', 'Arrive', 'Depart'},
    'Train-OfferBook': {'Time', 'Leave', 'Day', 'Dest', 'Ref', 'Ticket', 'none', 'Choice', 'People', 'Id', 'Arrive', 'Depart'},
    'Train-OfferBooked': {'Time', 'Leave', 'Day', 'Dest', 'Ref', 'Ticket', 'none', 'Choice', 'People', 'Id', 'Arrive', 'Depart'},
    'Train-Request': {'Leave', 'Day', 'Dest', 'People', 'Arrive', 'Depart'},
    'Train-Select': {'Leave', 'Day', 'Dest', 'Ticket', 'none', 'Choice', 'People', 'Id', 'Arrive', 'Depart'},
    'general-bye': {'none'},
    'general-greet': {'none'},
    'general-reqmore': {'none'},
    'general-welcome': {'none'}
}


def da_normalize(das, role):
    """
    normalize the output of NLU
    :param das: {act:[[slot,value],...],...}
    :param role: 'usr' or 'sys'
    :return:
    """
    if isinstance(das, str):
        return das
    new_das = deepcopy(das)
    for act, svs in das.items():
        if act not in valid_intent2slot[role]:
            new_das.pop(act)
            continue
        for s, v in svs:
            if s not in valid_intent2slot[role][act]:
                new_das[act].remove([s, v])
        if not new_das[act]:
            new_das.pop(act)
    return new_das


if __name__ == '__main__':
    das = da_normalize({
          "Attraction-Inform": [
            [
              "Price",
              "east"
            ],
            [
              "Area",
              "4"
            ]
          ]
        }, 'usr')
    print(das)
