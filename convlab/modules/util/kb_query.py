"""
"""
import random, copy

class KBQuery:
    def __init__(self, kb_path):
        self.kb_path = kb_path
        # configure knowledge base TODO: yaoqin

    def query(self, state):
        """
        Args:
            state (dict): The current state, every item is updated except 'kb_result_dict'.
                    state['current_slots']['inform_slots'] provides the constraints collected so far.
                    Example: {'price': 'expensive', 'location': 'north'}
        Returns:
            query_result (list): A list of query results, each item is a instance dict.
                    Example: [{'name': 'xxx hotel', 'price': expensive', 'location': 'north', ...},
                              {'name': 'xxxx hotel', ...}]
        """
        # implemented TODO: yaoqin

        # below is a trivial implementation by ramdomly return results
        # meta_result = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'south',
        #                'duration': 'xxx_time'}
        slot_set = ['fee', 'address', 'addr', 'area', 'stars', 'internet', 'department', 'choice', 'ref', 'food', 'type', 'price',
                    'pricerange', 'stay', 'phone', 'post', 'day', 'trainid', 'name', 'car', 'leave', 'time', 'arrive', 'ticket',
                    'none', 'depart', 'people', 'dest', 'parking', 'duration', 'open', 'id', 'entrance fee']
        meta_result = {}
        for slot in slot_set:
            meta_result[slot] = '{}-value'.format(slot)

        result_no = random.randint(1, 5)
        kb_result = []
        for idx in range(result_no):
            item = copy.deepcopy(meta_result)
            item['name'] = item['name'] + str(idx)
            kb_result.append(item)
        return kb_result