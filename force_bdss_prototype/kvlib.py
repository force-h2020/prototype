from kivy.app import App
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.properties import NumericProperty, BoundedNumericProperty, StringProperty, ListProperty, BooleanProperty

Builder.load_string('''
<Demo>:
    Label:
        text: "MinMaxSlider:"
    MinMaxSlider:
        min: 0
        max: 1
    Label:
        text: "AutoCenteredTextInput:"
    AutoCenteredTextInput:
        text: "Hello World"
    Label:
        text: "EditableLabel:"
    EditableLabel:
        text: "Double Click me"
    Label:
        text: "RotatableLabel:"
    RotatableLabel:
        text: "Hello World"
        angle: 180
    Label:
        text: "KeyboardListener:"
    KeyboardListenerDemo:
        text: "[]"
    
<MinMaxSlider>:
    MinSlider:
        id: min
        min: root.min
        max: root.max
        value: root.value0
        step: root.step
        sensitivity: 'handle'
        opacity: 0.75
        on_value: root.updateValue0(self.value)
    MaxSlider:
        id: max
        min: root.min
        max: root.max
        value: root.value1
        step: root.step
        background_width: 0
        sensitivity: 'handle'
        opacity: 0.75
        on_value: root.updateValue1(self.value)

<EditableLabel>:
    AutoCenteredTextInput:
        id: input
        opacity: 0
        text: root.text
        multiline: False
        on_text_validate: root.lock()
    Label:
        id: label
        text: root.text
        text_size: root.text_size
        on_touch_down: root.unlock(args[1])

<AutoCenteredTextInput>:
    line_heigth: 12
    multiline: False
    padding_x: (self.width - self._get_text_width(self.text, self.tab_width, self._label_cached))/2
    padding_y: self.height / 2.0 - self.line_heigth
    on_text: root.recenter()

<RotatableLabel>:
    canvas.before:
        PushMatrix
        Rotate:
            angle: self.angle
            origin: self.center
    canvas.after:
        PopMatrix
''')


class DemoApp(App):
    def build(self):
        return DemoScrollView()

class DemoScrollView(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.view = Demo(cols=2, size_hint=(1, None))
        self.view.bind(minimum_height=self.view.setter('height'))
        self.view.row_default_height = 100
        self.add_widget(self.view)

class Demo(GridLayout):
    pass

class MaxSlider(Slider):
    def on_touch_move(self, touch):
        if touch.grab_current == self:
            if touch.pos > self.parent.getMinPos():
                self.value_pos = touch.pos
            return True

    def on_touch_up(self, touch):
        if touch.grab_current == self:
            if touch.pos > self.parent.getMinPos():
                self.value_pos = touch.pos
            return True

class MinSlider(Slider):
    def on_touch_move(self, touch):
        if touch.grab_current == self:
            if touch.pos < self.parent.getMaxPos():
                self.value_pos = touch.pos
            return True
    
    def on_touch_up(self, touch):
        if touch.grab_current == self:
            if touch.pos < self.parent.getMaxPos():
                self.value_pos = touch.pos
            return True

class MinMaxSlider(AnchorLayout):
    min = NumericProperty(0.)
    max = NumericProperty(100.)
    value0 = NumericProperty(0.)
    value1 = NumericProperty(100.)
    step = BoundedNumericProperty(0, min=0)

    def getMinPos(self):
        return self.ids["min"].value_pos
    
    def getMaxPos(self):
        return self.ids["max"].value_pos

    def on_touch_down(self, touch):
        if self.disabled or not self.collide_point(*touch.pos):
            return
        if self.ids['min'].children[0].collide_point(*touch.pos):
            touch.grab(self.ids['min'])
        elif self.ids['max'].children[0].collide_point(*touch.pos):
            touch.grab(self.ids['max'])
        return True
    
    def updateValue0(self, value):
        self.value0 = value
    
    def updateValue1(self, value):
        self.value1 = value

class EditableLabel(AnchorLayout):
    text = StringProperty("default")
    textCache = StringProperty("default")
    text_size = ListProperty((None, None))
    
    def unlock(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_double_tap:
                self.ids["input"].opacity = 1
                self.ids["label"].opacity = 0
                self.textCache = self.text
    
    def lock(self):
        self.text = self.ids["input"].text
        self.ids["input"].opacity = 0
        self.ids["label"].opacity = 1

class AutoCenteredTextInput(TextInput):
    def recenter(self):
        self.padding_x =  (self.width - self._get_text_width(self.text, self.tab_width, self._label_cached))/2

class RotatableLabel(Label):
    angle = NumericProperty(90)

class KeyboardListener(Widget):
    pressedKeys = ListProperty()
    alwaysListen = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._request_keyboard()
    
    def _keyboard_closed(self):
        if(not self.alwaysListen):
            self._keyboard.unbind(on_key_down=self._on_keyboard_down)
            self._keyboard.unbind(on_key_up=self._on_keyboard_up)
            self._keyboard = None

    def _request_keyboard(self):
        self._keyboard = Window.request_keyboard(
            self._keyboard_closed, self, 'text')
        if(self._keyboard):
            self._keyboard.bind(on_key_down=self._on_keyboard_down)
            self._keyboard.bind(on_key_up=self._on_keyboard_up)

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if(not keycode in self.pressedKeys):
            self.pressedKeys.append(keycode)
        return True
    
    def _on_keyboard_up(self, keyboard, keycode):
        self.pressedKeys.remove(keycode)
        return True

class KeyboardListenerDemo(KeyboardListener, Label):
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if(not keycode in self.pressedKeys):
            self.pressedKeys.append(keycode)
        self.text = str(self.pressedKeys)
        return True
    
    def _on_keyboard_up(self, keyboard, keycode):
        self.pressedKeys.remove(keycode)
        self.text = str(self.pressedKeys)
        return True

class QuickPopup(Widget):
    def popup(self, title, content):
        popup = Popup(title=title,
        content=Label(text=content, text_size= (350, None)),
        size_hint=(None, None), size=(400, 300))
        popup.open()
        return

if __name__ == '__main__':
    DemoApp().run()

# Things to add:
#   + setSlider: Slider that operates on a set of values --> jumps only between values in the set
#   + 
