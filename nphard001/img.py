import datetime
now = lambda: datetime.datetime.now().timestamp()
class TestClass:
    def text(self):
        ts = now()
        return f'TestClass(text="welcome now {ts:.0f}")'
    def image(self):
        return f'TestClass(image=None)'
