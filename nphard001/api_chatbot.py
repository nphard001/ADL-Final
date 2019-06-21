from nphard001.api_data import *
class HostDataChatbotAPI(HostDataAPI):
    def get_pending_list(self, verbose=True)->list:
        r'''
        return dialogs need to reply by pending_list from linux7
        if exceptions like http occur, got empty list instead
        example:
        [
         {
         "line_userId": "U9d6503da59b6b0d5efba3c7af7af5125",
         "text_list": [
         "msg1",
         "msg2",
         "msg3"
         ],
         "token": "d40dc75af921458cbcddccef57c42b49"
         }
        ]
        '''
        r = HTTPJson(f'{self.host}/data/pending_list', {})
        self._r = r
        
        if r.status_code != 200:
            if verbose:
                print('HTTP code not ok', r, file=sys.stderr)
            return []
        
        try:
            j = r.json()
            self._j = j
            return j['pending_list']
        except ValueError as e:
            if verbose:
                print('json parse failed', str(e), file=sys.stderr)
            return []
    def send_reply_index(self, line_userId: str, img_idx: int, verbose=True):
        r'''
        send image index in train_im [0, 9999] to line_userId
        linux7 server solves avaliable reply_token from its database
        it is not guaranteed that LINE message API will accept it
        '''
        # # example
        # tosend = {
        #     'line_userId': 'U9d6503da59b6b0d5efba3c7af7af5125',
        #     'img_idx': 9999,
        # }
        tosend = {
            'line_userId': line_userId,
            'img_idx': img_idx,
        }
        j = HTTPJson2Json(f'{self.host}/data/reply_index', tosend)
        self._j = j
        return j
    
    @staticmethod
    def demo():
        # from nphard001.api_chatbot import *
        api = HostDataChatbotAPI()
        pending_list = api.get_pending_list()
        for pending in pending_list:
            print('pending:')
            print(json.dumps(pending, indent=1))
            num_msg = len(pending['text_list'])
            line_userId = pending['line_userId']
            img_idx = num_msg
            
            print('reply user', line_userId, 'with image index', img_idx)
            send_result = api.send_reply_index(line_userId, img_idx)
            print(json.dumps(send_result, indent=1))