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
#sympy
from sympy import sympify, symbols, Matrix, SympifyError
#kvlib
from .kvlib import EditableLabel


#kv
Builder.load_string('''
<FunctionsHeader>:
    size_hint_y: 0.1
    orientation: 'horizontal'
    Label:
        text: 'id'
    Label:
        text: 'Description'
        text_size: self.width, None
        size_hint_x: 1.25
    Label:
        text: 'Function'
        size_hint_x: 4

<AttributesHeader>:
    size_hint_y: 0.2
    orientation: 'horizontal'
    Label:
        text: 'Attribute'
    Label:
        text: 'Description'

<ParameterHeader>:
    size_hint_y: 0.2
    orientation: 'horizontal'
    Label:
        text: 'Parameter'
    Label:
        text: 'Description'

<FunctionBottomRow>:
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
    Label:
        text: root.description
        text_size: self.width, None
        size_hint_x: root.layout[1]
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
        editor = BoxLayout(orientation = "horizontal", size_hint_y = 4)
        self.functionWrapper = FunctionWrapperWidget(self.input[0], do_scroll_x=False)
        editor.add_widget(WrapperWithHeader(FunctionsHeader(), self.functionWrapper))
        
        varWrapper = BoxLayout(orientation = "vertical", size_hint_x = 0.5)
        self.attributes = VarWrapperWidget(self.input[1], do_scroll_x=False)
        self.parameters = VarWrapperWidget(self.input[2], do_scroll_x=False)
        varWrapper.add_widget(WrapperWithHeader(AttributesHeader(), self.attributes))
        varWrapper.add_widget(WrapperWithHeader(ParameterHeader(), self.parameters))
        
        #editor.add_widget(self.functionWrapper)
        editor.add_widget(varWrapper)
        root.add_widget(editor)

        root.add_widget(FunctionBottomRow())
        return root
    
    #sympifies the given function and checks if all contained symbols are initialized variables or
    #sub functions. checks for direct recursive calls
    def validate(self, func, var_set, func_set):
        out = [sympify("0"), ""]
        func_desc = self.functionWrapper.getFunctionWidgets()[func].getDescription()
        #if func_set[func] == "default": return out # @default this enables "default" as valid input
        try:
            function = sympify(func_set[func])
        except SympifyError:
            out[1] = func_desc + " contains a syntax error" 
            return out
        for var in function.free_symbols:
            if var in var_set: continue
            elif var == func:
                out[1] = func_desc + " contains a direct recursive call" 
                return out
            elif var in func_set: continue
            else: 
                out[1] = func_desc + " contains unknown identifier: " + str(var) 
                return out
        out[0] = function
        return out

    #validates all functions and checks for recursive functions calls + error handling 
    def validateAll(self):
        errors = []
        var_set = self.attributes.getVariables() + self.parameters.getVariables()
        func_set = self.functionWrapper.getFunctions()
        temp = {}
        for func in func_set:
            out = self.validate(func, var_set, func_set)
            temp.update({func: out[0]})
            if(out[1] != ""):
                errors.append(out[1])
        func_set = temp # func_set now contains sympified functions which can be test evaluated
        for func in func_set:
            function = func_set[func]
            #try to substitute all subfunctions
            try:
                #if(function == "default"): continue @default
                recursion_counter = 0
                max_recursion_depth = 10000
                while len(function.free_symbols.intersection(func_set)) != 0:
                    sub_func = function.free_symbols.intersection(func_set).pop()
                    function = function.subs(sub_func, func_set[sub_func])
                    recursion_counter += 1
                    if recursion_counter > max_recursion_depth: raise RecursionError()
            except RecursionError:
                func_desc = self.functionWrapper.getFunctionWidgets()[func].getDescription()
                message = func_desc + " exceeded the max-recursion depth, there might be a recursive call." 
                errors.append(message)       
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

        #remove sub-functions from func_set + func derivatives
        attributes = self.attributes.getVariables()
        print(attributes)
        AO = []

        for func in func_set:
            if self.functionWrapper.getFunctionWidgets()[func].isEditable(): continue
            function = func_set[func]
            while len(function.free_symbols.intersection(func_set)) != 0:
                    sub_func = function.free_symbols.intersection(func_set).pop()
                    function = function.subs(sub_func, func_set[sub_func])
            # func derivatives with KPIs
            # create output + add original function

            AO.append(function)

        grad_A_AO = []

        for func in AO:
            temp = attributes[:]
            i = 0
            for a in attributes:
                temp[i] = func.diff(a) 
                i += 1
            grad_A_AO += [temp]
                
        self.output = (Matrix(AO),Matrix(grad_A_AO).transpose()) 
        return True
    
    #adds a new editable function #TODO: no name duplications
    def addFunction(self):
        self.functionWrapper.addEditableFunction("id","desc","1")
    
    #runs the application and returns its output
    def run_with_output(self, editorInput, defaultOutput):
        self.input = editorInput
        self.output = defaultOutput
        self.run()
        return self.output

    #stops the program and saves its output if no function contains an error
    def stop_with_output(self):
        if self.validateAll():
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
        output = {}
        for func in self.functionWidgets:
            output.update({symbols(func.getID()): func.getFunction()})
        return output
    
    def getFunctionWidgets(self):
        output = {}
        for func in self.functionWidgets:
            output.update({symbols(func.getID()): func})
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
    
    def updateFunction(self, newFunction):
        self.function = newFunction

    def validate(self):
        pass

    def isEditable(self):
        pass

class EditableFunctionWidget(FunctionWidgetBase):
    def updateID(self, newID):
        #TODO: no name duplication
        self.identifier = newID

    def updateDesc(self, newDesc):
        self.description = newDesc

    def isEditable(self):
        return True

class StaticFunctionWidget(FunctionWidgetBase):
    def isEditable(self):
        return False

class VarWrapperWidget(ScrollView):
    def __init__(self, variables, **kwargs):
        super().__init__(**kwargs)
        self.view = GridLayout(cols=1, size_hint=(1, None))
        self.view.bind(minimum_height=self.view.setter('height'))
        self.variableWidgets = []
        for key, value in variables.items():
            self.variableWidgets.append(VarWidget(key, value,size_hint=(1,None), size = (1,25)))
            self.view.add_widget(self.variableWidgets[len(self.variableWidgets)-1])
        self.add_widget(self.view)
    
    def getVariables(self):
        output = []
        for var in self.variableWidgets:
            output.append(symbols(var.getVariable()))
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

class WrapperWithHeader(BoxLayout):
    orientation = "vertical"

    def __init__(self, header, content, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(header)
        self.add_widget(content)

class FunctionsHeader(BoxLayout): pass
class AttributesHeader(BoxLayout): pass
class ParameterHeader(BoxLayout): pass
class FunctionBottomRow(BoxLayout): pass


######### Main  #########
if __name__ == '__main__':
    #functions: key: id, value[0]: description, value[1]: function, value[2]: isEditable
    functions = {
        "pc" : ["Production Cost","t * (T - 290)^2 * W", False],
        "mc" : ["Material Cost" , "mcA + mcB", False],
        "mcA" : ["Mat cost A" , "(cost_purification * (C_e / C_supplier -1)^2 + const_A) * V_a + V_r * quad_coeff * (V_a - 0.6 * V_r)**2", True],
        "mcB" : ["Mat cost B" , "(V_r - V_a) * cost_B", True],
        "imp" : ["Impurity Concentration" , "ln((conc_A + conc_B + conc_C + conc_S )/ C_supplier)", False]
    }
    
    attributes = {
        "V_a" : "volume of A",
        "C_e" : "impurity of A",
        "T" : "temperature",
        "t" : "reaction time",
        "conc_A" : "concentration of A",
        "conc_B" : "concentration of B",
        "conc_P" : "concentration of P",
        "conc_S" : "concentration of S",
        "conc_C" : "concentration of C"
    }
    
    fixed_parameters = {
        "p_A" : "pure density of A",
        "p_B" : "pure density of A",
        "p_C" : "pure density of A",
        "V_r" : "reactor volume",
        "W" : "heating cost",
        "const_A" : "description",
        "cost_B" : "description",
        "quad_coeff" : "description",
        "C_supplier" : "description",
        "cost_purification" : "description"
    }
    
    editorInput = [functions, attributes, fixed_parameters]
    print(FunctionApp().run_with_output(editorInput, "Default"))
