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
from sympy import sympify, symbols, Matrix
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

        root.add_widget(TopRow())

        editor = BoxLayout(orientation = "horizontal", size_hint_y = 4)
        self.functionWrapper = FunctionWrapperWidget(self.input[0], do_scroll_x=False)
        self.varWrapper = VarWrapperWidget(self.input[1], size_hint_x = 0.5)
        editor.add_widget(self.functionWrapper)
        editor.add_widget(self.varWrapper)
        root.add_widget(editor)

        root.add_widget(BottomRow())
        return root
    
    #sympifies the given function and checks if all contained symbols are initialized variables or
    #sub functions. checks for direct recursive calls
    def validate(self, func, var_set, func_set):
        out = ["default", ""]
        if func_set[func] == "default": return out # @default this enables "default" as valid input
        function = sympify(func_set[func])
        for var in function.free_symbols:
            if var in var_set: continue
            elif var == func: 
                func_desc = self.functionWrapper.getFunctionWidgets()[func].getDescription() 
                out[1] = func_desc + " contains a direct recursive call" 
                return out
            elif var in func_set: continue
            else: 
                func_desc = self.functionWrapper.getFunctionWidgets()[func].getDescription() 
                out[1] = func_desc + " contains unknown identifier: " + str(var) 
                return out
        out[0] = function
        return out

    #validates all functions and checks for recursive functions calls + error handling 
    def validateAll(self):
        errors = []
        var_set = self.varWrapper.getVariables()
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
                if(function == "default"): continue
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
            add=Label(text=content, text_size= (350, None)),
            size_hint=(None, None), size=(400, 300))
            popup.open()
            return False

        #remove sub-functions from func_set + func derivatives
        self.output = {}
        x_set = self.varWrapper.getXDimension()
        y_set = self.varWrapper.getYDimension()

        for func in func_set:
            if self.functionWrapper.getFunctionWidgets()[func].isEditable(): continue
            function = func_set[func]
            if function == "default": 
                #TODO: default handle when derivatives are implemented
                self.output.update({func: "default"})
                continue
            #remove sub-functions, already checked for recursion errors 
            while len(function.free_symbols.intersection(func_set)) != 0:
                    sub_func = function.free_symbols.intersection(func_set).pop()
                    function = function.subs(sub_func, func_set[sub_func])
            # func derivatives with KPIs
            # create output + add original function

            func_out = []
            #determine if function is part of X-dimension
            inX = False
            for var in function.free_symbols:
                if var in x_set:
                    inX = True
                
            if(inX):
                for x in x_set:
                    func_out.append(function.diff(x))
                #TODO: find better solution for T & tau which are both in X & y    
                func_out.append(function.diff(symbols("T")))
                func_out.append(function.diff(symbols("tau")))            
            else:
                for y in y_set:
                    func_out.append(function.diff(y))
            self.output.update({func: (function ,Matrix(func_out))})            
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

class VarWrapperWidget(StackLayout):
    variableWidgets = []

    def __init__(self, variables, **kwargs):
        super().__init__(**kwargs)
        for key, value in variables.items():
            self.variableWidgets.append(VarWidget(key, value[0], value[1], size_hint_y=1/len(variables)))
            self.add_widget(self.variableWidgets[len(self.variableWidgets)-1])
    
    def getVariables(self):
        output = {}
        for var in self.variableWidgets:
            output.update({symbols(var.getVariable()): var.getDesignDimension()})
        return output
    
    def getYDimension(self):
        var_set = self.getVariables()
        y_set = []
        for var in var_set:
            if var_set[var] == "y":
                y_set.append(var)
        return y_set
    
    def getXDimension(self):
        var_set = self.getVariables()
        X_set = []
        for var in var_set:
            if var_set[var] == "X":
                X_set.append(var)
        return X_set

            

class VarWidget(BoxLayout):
    name = StringProperty()
    description = StringProperty()

    def __init__(self, name, description, designDimension , **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.description = description
        self.designDimension = designDimension

    def getVariable(self):
        return self.name
    
    def getDesignDimension(self):
        return self.designDimension

class TopRow(BoxLayout): pass
class BottomRow(BoxLayout): pass


######### Main  #########
if __name__ == '__main__':
    #functions: key: id, value[0]: description, value[1]: function, value[2]: isEditable
    functions = {"pc" : ["Production Cost","tau * (T - 290)^2 * W", False],
                 "mc" : ["Material Cost" , "mcA + mcB", False],
                 "mcA" : ["Mat cost A" , "(cost_p * (C_e / C_sup -1)^2 + const_A) * V_a", True],
                 "mcB" : ["Mat cost B" , "(V_r - V_a) * cost_B", True],
                 "imp" : ["Impurity Concentration" , "conc_A + conc_B + conc_C + conc_S", False]}
    #variables: key: id, value[0]: description, value[1]: isFixedParameter
    var = {"V_a" : ("volume of A","y"),
           "C_e" : ("impurity of A","y"),
           "T" : ("temperature","y"),
           "tau" : ("reaction time","y"),
           "conc_A" : ("concentration of A","X"),
           "conc_B" : ("concentration of B","X"),
           "conc_P" : ("concentration of P","X"),
           "conc_S" : ("concentration of S","X"),
           "conc_C" : ("concentration of C","X"),
           "W" : ("heating cost","fixed"),
           "C_sup" : ("description","fixed"),
           "cost_p" : ("description","fixed"),
           "const_A" : ("description","fixed"),
           "cost_B" : ("description","fixed"),
           "V_r" : ("reactor volume","fixed"),}
    editorInput = [functions, var]
    print(FunctionApp().run_with_output(editorInput, "Default"))
