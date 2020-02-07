from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.lang import Builder
#Propertys Import
from kivy.properties import StringProperty, ListProperty
#Layout Import
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.scrollview import ScrollView
#Interactable Import
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
#Function Import
from pyparsing import ParseException
from function_parser import parseFunction, evaluateStack
import math
#kvlib
from kvlib import EditableLabel

#kv
Builder.load_string('''
<TopRow>:
    size_hint_y: 0.5
    BoxLayout:
        orientation: 'horizontal'
        size_hint_x: 2
        Label:
            text: 'id'
        Label:
            text: 'Description'
            text_size: self.width, None
            size_hint_x: 1.25
        Label:
            text: 'Function'
            size_hint_x: 4
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'Variable'
        Label:
            text: 'Description'

<BottomRow>:
    size_hint_y: 0.5
    Button:
        text: 'Add Function'
        on_release: app.addFunction()
    Button:
        text: 'Validate'
        on_release: app.validateAll()
    Button:
        text: 'Submit'
        on_release: app.stop_with_output()

<EditableFunctionWidget>:
    EditableLabel:
        text: root.identifier
        size_hint_x: root.layout[0]
        on_text: root.updateID(self.text)
    EditableLabel:
        text: root.description
        text_size: self.width, None
        size_hint_x: root.layout[1]
        on_text: root.updateDesc(self.text)
    TextInput:
        text: root.function
        multiline: False
        size_hint_x: root.layout[2]
        padding_y: self.height / 2.0 - 12
        on_text: root.updateFunction(self.text)

<StaticFunctionWidget>:
    Label:
        text: root.identifier
        size_hint_x: root.layout[0]
        on_text: root.updateID(self.text)
    Label:
        text: root.description
        text_size: self.width, None
        size_hint_x: root.layout[1]
        on_text: root.updateDesc(self.text)
    TextInput:
        text: root.function
        multiline: False
        size_hint_x: root.layout[2]
        padding_y: self.height / 2.0 - 12
        on_text: root.updateFunction(self.text)

<VarWidget>:
    orientation: 'horizontal'
    Label:
        text: root.name
    Label:
        text: root.description
''')


#window
class FunctionApp(App):
    output = []

    def build(self):
        #input: [(Set of function names),(Set of parameter names)]
        root = BoxLayout(orientation = "vertical")

        root.add_widget(TopRow())

        editor = BoxLayout(orientation = "horizontal", size_hint_y = 4)
        self.functionWrapper = FunctionWrapperWidget(self.input[0], do_scroll_x=False)
        self.varWrapper = VarWrapperWidget(self.input[1], size_hint_x = 0.5)
        editor.add_widget(self.functionWrapper)
        editor.add_widget(self.varWrapper)
        root.add_widget(editor)

        root.add_widget(BottomRow())
        return root
    
    def validate(self, function):
        out = ["default", ""]
        fn  = ["sin", "cos", "tan", "exp", "abs", "trunc", "round", "sgn", "ln", "log2", "log10"]
        var_set = self.varWrapper.getVariables()
        func_set = self.functionWrapper.getFunctionIDs()
        try:
            out[0] = parseFunction(function.function)
        except ParseException:
            out[1] = str(function.description) + " contains an error"
        else:
        #no parsing error --> check if there are no direct recursive calls and unknown identifier  
        #--> parsing is correct 
            if(out[0] == ["default"]): return out
            for op in out[0]:
                if op == 'unary -':
                    continue
                if op in "+-*/^":
                    continue
                elif op == "PI":
                    continue
                elif op == "E":
                    continue
                elif op == "default":
                    continue
                elif op in fn:
                    continue
                elif op in func_set:
                    if (op == function.identifier): return [["invalid"], str(function.description) + " contains a direct recursive call"]
                    else: continue
                elif op in var_set:
                    continue
                elif op[0].isalpha():
                    return [["invalid"], str(function.description) + "contains unknown identifier '%s'" % op]    
                else:
                    continue
        return out
        
    def validateAll(self):
        output = []
        errors = []
        for func in self.functionWrapper.getFunctions():
            out = self.validate(func)
            output.append(out[0])
            if(out[1] != ""):
                errors.append(out[1])
        #Test Eval to check if max-recursion depth is exceded
        var_set = self.varWrapper.getVariables()
        func_IDs = self.functionWrapper.getFunctionIDs()
        func_set = {}
        for i in range(len(func_IDs)):
            func_set.update({func_IDs[i] : output[i]})
        #print(output)
        for i in range(len(output)):
            func = output[i]
            try: 
                if(func[0] == "default" or func[0] == "invalid"): continue
                evaluateStack(func[:], var_set, func_set)
            except Exception as e:
                name = self.functionWrapper.getFunctions()[i].getDescription()
                errors.append(str(name) + " failed evaluation test: " + str(e))
        #Error handling
        if(len(errors)!=0): 
            if(len(errors) == 1): content = "There is 1 problem: \n"
            else: content = "There are " + str(len(errors)) + " problems: \n"
            for error in errors: content += error + "\n"
            self.output = "invalid"
            popup = Popup(title='Warning',
            content=Label(text=content, text_size= (350, None)),
            size_hint=(None, None), size=(400, 300))
            popup.open()
            return False
        self.output = output
        return True
    
    def addFunction(self):
        self.functionWrapper.addEditableFunction("id","desc","1")
    
    def run_with_output(self, editorInput, defaultOutput):
        self.input = editorInput
        self.output = defaultOutput
        self.run()
        return self.output

    def stop_with_output(self):
        if self.validateAll():
            #replace function-calls
            func_IDs = self.functionWrapper.getFunctionIDs()
            for i in range(len(self.output)):
                func = self.output[i]
                for ID in func_IDs:
                    while(ID in func):
                        print(func, ID)
                        index = func.index(ID)
                        func = func[:index] + self.output[func_IDs.index(ID)] + func[index+1:]
                self.output[i] = func
            #remove non-relevant functions
            out = []
            functions = self.functionWrapper.getFunctions()
            for i in range(len(self.output)):
                if(not functions[i].isEditable()):
                    out.append({"name": functions[i].getDescription(), "function": self.output[i]})
            self.output = out
            self.stop()

#Widgets
class FunctionWrapperWidget(ScrollView):
    functionWidgets = []

    def __init__(self, functions, **kwargs):
        super().__init__(**kwargs)
        self.view = GridLayout(cols=1, size_hint=(1, None))
        self.view.bind(minimum_height=self.view.setter('height'))
        for identifier in functions:
            if(functions[identifier][2]):
                self.addEditableFunction(identifier, functions[identifier][0], functions[identifier][1])
            else:
                self.addStaticFunction(identifier, functions[identifier][0], functions[identifier][1])
        self.add_widget(self.view)

    def addEditableFunction(self, identifier, description, function):
        self.functionWidgets.append(EditableFunctionWidget(identifier, description, function, size_hint=(1,None), size = (50,100)))
        self.view.add_widget(self.functionWidgets[len(self.functionWidgets)-1])
    
    def addStaticFunction(self, identifier, description, function):
        self.functionWidgets.append(StaticFunctionWidget(identifier, description, function, size_hint=(1,None), size = (50,100)))
        self.view.add_widget(self.functionWidgets[len(self.functionWidgets)-1])
        
    def getFunctions(self):
        output = []
        for func in self.functionWidgets:
            output.append(func)
        return output
    
    def getFunctionIDs(self):
        output = []
        for func in self.functionWidgets:
            output.append(func.getID())
        return output

class FunctionWidgetBase(BoxLayout):
    layout = (1,1.25,4)

    identifier = StringProperty()
    description = StringProperty()
    function = StringProperty()

    def __init__(self, identifier, description, function, **kwargs):
        super().__init__(**kwargs)
        self.identifier = identifier
        self.description = description
        self.function = function

    def getID(self):
        return self.identifier

    def getDescription(self):
        return self.description

    def getFunction(self):
        return self.function

    def updateID(self, newID):
        self.identifier = newID

    def updateDesc(self, newDesc):
        self.description = newDesc

    def updateFunction(self, newFunction):
        self.function = newFunction

    def validate(self):
        pass

    def isEditable(self):
        pass

class EditableFunctionWidget(FunctionWidgetBase):
    def isEditable(self):
        return True

class StaticFunctionWidget(FunctionWidgetBase):
    def isEditable(self):
        return False

class VarWrapperWidget(StackLayout):
    variableWidgets = []

    def __init__(self, variables, **kwargs):
        super().__init__(**kwargs)
        for key, value in variables.items():
            self.variableWidgets.append(VarWidget(name=key, description=value, size_hint_y=1/len(variables)))
            self.add_widget(self.variableWidgets[len(self.variableWidgets)-1])
    
    def getVariables(self):
        output = {}
        for var in self.variableWidgets:
            output.update({var.getVariable():1})
        return output
            

class VarWidget(BoxLayout):
    name = StringProperty()
    description = StringProperty()

    def __init__(self, name, description, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.description = description

    def getVariable(self):
        return self.name

class TopRow(BoxLayout): pass
class BottomRow(BoxLayout): pass


######### Main  #########
if __name__ == '__main__':
    functions = {"pc" : ["Production Cost","tau * (T - 290)^2 * W", False],
                 "mc" : ["Material Cost" , "mcA + mcB", False],
                 "mcA" : ["Mat cost A" , "(cost_p * (C_e / C_sup -1)^2 + const_A) * V_a", True],
                 "mcB" : ["Mat cost B" , "(V_r - V_a) * cost_B", True],
                 "imp" : ["Impurity Concentration" , "default", False]}
    var = {"V0" : "volume",
           "C_e" : "impurity of A",
           "tau" : "reaction time",
           "T" : "temperature",
           "W" : "heating cost",
           "C_sup" : "description",
           "cost_p" : "description",
           "const_A" : "description",
           "cost_B" : "description",
           "V_r" : "reactor volume",
           "V_a" : "volume of A"}
    editorInput = [functions, var]
    print(FunctionApp().run_with_output(editorInput, "Default"))
