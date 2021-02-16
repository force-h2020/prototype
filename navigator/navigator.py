from kivy import Config
Config.set('graphics', 'multisamples', '0')
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.stacklayout import StackLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import (NumericProperty, AliasProperty,
                             ObjectProperty, ListProperty)
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.core.window import Window
Window.maximize()
from kivy.utils import platform
if platform == "android":
    Window.release_all_keyboards()
from kivy.graphics import Color, Rectangle, Point, Ellipse
from kivy.garden.graph import Graph, Plot
import numpy as np

def search_all(valuearray, paramarray, unrestr):
    res = []
    for i in unrestr:
        bl = True
        for j in range(len(valuearray)):
            if valuearray[j] != paramarray[j, i]:
                bl = False
        if bl:
            res.append(i)
    return res

def fuse_arrays(ar1, ar2):
    mx = ar1.size
    a = np.zeros([mx, 2], dtype=np.float)
    for i in range(mx):
        a[i][0] = ar1[i]
        a[i][1] = ar2[i]
    return a


class MyGraph(Graph):

    picker_tol = 20

    def afuncx(self):
        return lambda x: 10**x if self.xlog else lambda x: x

    def afuncy(self):
        return lambda y: 10**y if self.ylog else lambda y: y

    def funcx(self):
        return np.log10 if self.xlog else lambda x: x

    def funcy(self):
        return np.log10 if self.ylog else lambda y: y

    def px_to_x(self, px):
        afuncx = self.afuncx()
        size = self.plots[0].params["size"]
        xmin = self.xmin
        xmax = self.xmax
        ratiox = (size[2] - size[0]) / float(xmax - xmin)
        x = (float((px - size[0]) / ratiox + xmin))
        return x

    def px_to_y(self, px):
        afuncy = self.afuncy()
        size = self.plots[0].params["size"]
        ymin = self.ymin
        ymax = self.ymax
        ratioy = (size[3] - size[1]) / float(ymax - ymin)
        y = (float((px - size[1]) / ratioy + ymin))
        return y

    def touched(self, x1, y1):
        size = self.plots[0].params["size"]
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        ratiox = (size[2] - size[0]) / float(xmax - xmin)
        ratioy = (size[3] - size[1]) / float(ymax - ymin)
        x = self.px_to_x(x1 - self.pos[0])
        y = self.px_to_y(y1 - self.pos[1])
        points = self.plots[1].points
        if len(points) < 2:
            points = self.plots[0].points
        p = np.zeros((len(points), 2))
        for i in range(len(points)):
            p[i, 0] = points[i][0]
            p[i, 1] = points[i][1]
        lx_min, lx_max = self.xmin, self.xmax
        lx = lx_max - lx_min
        ly_min, ly_max = self.ymin, self.ymax
        ly = ly_max - ly_min
        distances = np.hypot((x - p[:, 0]) * ratiox,
                             (y - p[:, 1]) * ratioy)
        indmin = distances.argmin()
        if distances[indmin] < self.picker_tol:
            return indmin
        else:
            return None

    def _redraw_size(self, *args):
        # size a 4-tuple describing the bounding box in which we can draw
        # graphs, it's (x0, y0, x1, y1), which correspond with the bottom left
        # and top right corner locations, respectively
        self._clear_buffer()
        size = self._update_labels()
        self.view_pos = self._plot_area.pos = (size[0], size[1])
        self.view_size = self._plot_area.size = (size[2] - size[0], size[3] - size[1])

        if self.size[0] and self.size[1]:
            self._fbo.size = self.size
        else:
            self._fbo.size = 1, 1  # gl errors otherwise
        self._fbo_rect.texture = self._fbo.texture
        self._fbo_rect.size = self.size
        self._fbo_rect.pos = self.pos
        self._background_rect.size = self.size
        self._update_ticks(size)
        self._update_plots(size)
        for plot in self.plots:
            plot.draw()


class DotPlot(Plot):

    def create_drawings(self):
        self._color = Color(*self.color)
        self._mesh = Point(points=(0, 0), pointsize=5)
        self.bind(color=lambda instr, value: setattr(self._color.rgba, value))
        return [self._color, self._mesh]

    def draw(self, *args):
        points = self.points
        mesh = self._mesh
        params = self._params
        funcx = log10 if params['xlog'] else lambda x: x
        funcy = log10 if params['ylog'] else lambda x: x
        xmin = funcx(params['xmin'])
        ymin = funcy(params['ymin'])
        size = params['size']
        ratiox = (size[2] - size[0]) / float(funcx(params['xmax']) - xmin)
        ratioy = (size[3] - size[1]) / float(funcy(params['ymax']) - ymin)
        mesh.points = ()
        for k in range(len(points)):
            x = (funcx(points[k][0]) - xmin) * ratiox + size[0]
            y = (funcy(points[k][1]) - ymin) * ratioy + size[1]
            mesh.add_point(x, y)

    def _set_pointsize(self, value):
        if hasattr(self, '_mesh'):
            self._mesh.pointsize = value
    pointsize = AliasProperty(lambda self: self._mesh.pointsize, _set_pointsize)

    def _set_source(self, value):
        if hasattr(self, '_mesh'):
            self._mesh.source = value
    source = AliasProperty(lambda self: self._mesh.source, _set_source)


# Slider widget
class MySlider(Slider):
    bot_restr = ObjectProperty(None)
    top_restr = ObjectProperty(None)
    bot_val = NumericProperty(0)
    top_val = NumericProperty(0)
    grabbed = "slider"
    label_val = NumericProperty(0)
    data = np.zeros(1)
    grey = ObjectProperty(np.zeros(1), force_dispatch=True)


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_val = self.value
        self.bot_val = 0 * (self.max - self.min) + self.min
        self.top_val = 1 * (self.max - self.min) + self.min
        bot_pos = self.val_into_pos_grey(self.bot_val) - 5 + self.x
        top_pos = self.val_into_pos_grey(self.top_val) + 1 + self.x
        self.bot_restr = Widget(pos=(bot_pos, self.center_y - 16), size=(4, 32))
        self.top_restr = Widget(pos=(top_pos, self.center_y - 16), size=(4, 32))
        with self.bot_restr.canvas:
            Color(1., 1., 1., 1.)
            self.bot_restr_canvas = Rectangle(pos=self.bot_restr.pos,
                                              size=self.bot_restr.size)
        with self.top_restr.canvas:
            Color(1., 1., 1., 1.)
            self.top_restr_canvas = Rectangle(pos=self.top_restr.pos,
                                              size=self.top_restr.size)
        restr_size = (int(max(self.size[0] / 100, 8)), int(self.size[1] / 1.5))
        self.bot_restr_canvas.size = restr_size
        self.top_restr_canvas.size = restr_size
        self.bot_restr.size = restr_size
        self.top_restr.size = restr_size
        self.bind(pos=self.pos_callback)
        self.bot_restr.bind(on_touch_move=self.restr_move_callback_b)
        self.top_restr.bind(on_touch_move=self.restr_move_callback_t)
        self.bot_restr.bind(pos=self.restr_pos_callback)
        self.top_restr.bind(pos=self.restr_pos_callback)
        self.bind(bot_val=self.bot_val_callback)
        self.bind(top_val=self.top_val_callback)
        self.add_widget(self.bot_restr)
        self.add_widget(self.top_restr)

    def restr_pos_callback(self, obj, val):
        self.bot_restr_canvas.pos = self.bot_restr.pos
        self.top_restr_canvas.pos = self.top_restr.pos

    def bot_val_callback(self, obj, val):
        bot_pos = self.val_into_pos_grey(self.bot_val) - self.bot_restr_canvas.size[0] - 1
        self.bot_restr.pos = (bot_pos, self.center_y - self.bot_restr_canvas.size[1] // 2)

    def top_val_callback(self, obj, val):
        top_pos = self.val_into_pos_grey(self.top_val) + 1
        self.top_restr.pos = (top_pos, self.center_y - self.top_restr_canvas.size[1] // 2)

    def pos_callback(self, obj, pos):
        bot_pos = self.val_into_pos_grey(self.bot_val) - self.bot_restr_canvas.size[0] - 1
        top_pos = self.val_into_pos_grey(self.top_val) + 1
        self.bot_restr.pos = (bot_pos, self.center_y - self.bot_restr_canvas.size[1] // 2)
        self.top_restr.pos = (top_pos, self.center_y - self.bot_restr_canvas.size[1] // 2)

    def restr_move_callback_b(touch, obj, value):
        self.bot_val = self.pos_into_val_grey(touch.pos)

    def restr_move_callback_t(touch, obj, value):
        self.top_val = self.pos_into_val_grey(touch.pos)

    # Convert a value into a position.
    # Used by: set_value_pos,
    #          PLayout: on_pick, change_two_sliders,
    #                   up_with_keys, down_with_keys
    def val_into_pos(self, val, data=np.array([])):
        if data.size == 0:
            data = self.data
        if not np.all(data == 0):
            val = float(data[(abs(data - val)).argmin()])
        x = self.x
        padding = self.padding
        width = self.width
        tmp = (width - 2*padding) * (val-self.min)
        if(self.max != self.min):
            return x + padding + tmp/(self.max-self.min)
        return x + padding + (width - 2*padding)


    # Convert a position into a value.
    # Used by: set_value_pos, on_touch_move
    def pos_into_val(self, pos, data=np.array([])):
        if data.size == 0:
            data = self.data
        x = self.x
        d = self.max - self.min
        padding = self.padding
        width = self.width
        val = (pos - x - padding)*d / (width - 2*padding) + self.min
        if not np.all(data == 0):
            val = float(data[(abs(data - val)).argmin()])
        if self.step:
            return min(round((val-self.min) / self.step)*self.step + self.min,
                       self.max)
        else:
            return val

    def val_into_pos_grey(self, val):
        if not np.all(self.grey == 0):
            val = float(self.grey[(abs(self.grey - val)).argmin()])
        x = self.x
        padding = self.padding
        width = self.width
        tmp = (width - 2*padding) * (val-self.min)
        if(self.max != self.min):
            return x + padding + tmp/(self.max-self.min)
        return x + padding + (width - 2*padding)

    def pos_into_val_grey(self, pos):
        x = self.x
        d = self.max - self.min
        padding = self.padding
        width = self.width
        val = (pos - x - padding)*d / (width - 2*padding) + self.min
        if not np.all(self.grey == 0):
            val = float(self.grey[(abs(self.grey - val)).argmin()])
        if self.step:
            return min(round((val-self.min) / self.step)*self.step + self.min,
                       self.max)
        else:
            return val

    # Get the position of the current value_normalized.
    # Used by: value_pos
    def get_value_pos(self):
        padding = self.padding
        x = self.x
        y = self.y
        nval = self.value_normalized
        if self.orientation == 'horizontal':
            return (x + padding + nval*(self.width - 2*padding), y)
        else:
            return (x, y + padding + nval*(self.height - 2*padding))

    # Set the current value with a given position.
    # Used by: value_pos
    def set_value_pos(self, pos):
        padding = self.padding
        bpos = self.val_into_pos(self.bot_val)
        tpos = self.val_into_pos(self.top_val)
        x = min(tpos, max(pos[0], bpos))
        self.value = self.pos_into_val(x)
        self.label_val = self.value

    value_pos = AliasProperty(get_value_pos, set_value_pos,
                              bind=('x', 'y', 'width', 'height', 'min',
                                    'max', 'value_normalized', 'orientation'))

    # Grab the restrictors, scroll or set the value to where the touch
    # is going down.
    # Used by: called if a touch is going down onto the slider
    def on_touch_down(self, touch):
        if self.disabled or not self.collide_point(*touch.pos):
            return
        if touch.is_mouse_scrolling:
            self.scroll_slider(touch)
        else:
            if self.bot_restr.collide_point(*touch.pos):
                touch.grab(self)
                self.grabbed = "bot_val"
                self.label_val = self.bot_val
                return True
            if self.top_restr.collide_point(*touch.pos):
                touch.grab(self)
                self.grabbed = "top_val"
                self.label_val = self.top_val
                return True
            touch.grab(self)
            self.grabbed = "slider"
            self.value_pos = touch.pos
            self.label_val = self.value
        return True

    # Scroll slider.
    # Used by: on_touch_down
    def scroll_slider(self, touch):
        curr_ind = (abs(self.data - self.value)).argmin()
        if 'down' in touch.button or 'left' in touch.button:
            self.value = min(self.top_val,
                float(self.data[min(curr_ind + 1, self.data.size - 1)]))
        if 'up' in touch.button or 'right' in touch.button:
            self.value = max(self.bot_val,
                float(self.data[max(curr_ind - 1, 0)]))
        self.label_val = self.value

    # If bot_restr is grabbed move it within min of slider and top_restr.
    # If top_restr is grabbed move it within max of slider and bot_restr.
    # If needed changes value so it is between bot_restr and top_restr.
    # If no restrictor is grabbed move value.
    # Used by: called if a touch is moving
    def on_touch_move(self, touch):
        if touch.grab_current == self:
            if self.grabbed == "slider":
                self.value_pos = touch.pos
                self.label_val = self.value
            else:
                if self.grabbed == "bot_val":
                    d = 1
                    border1 = self.min
                    border2 = self.top_val
                    #border2 = self.data[np.argmin(self.data - self.top_val) - 1]
                else:
                    d = -1
                    border1 = self.max
                    border2 = self.bot_val
                    #border2 = self.data[np.argmin(self.data - self.bot_val) + 1]
                if d * (self.pos_into_val_grey(touch.x)-border1) < 0:
                    setattr(self, self.grabbed, border1)
                elif d * (self.pos_into_val_grey(touch.x)-self.value) > 0:
                    if self.data.size > 1:
                        val = self.data[self.get_data_idx() + d]
                        self.value_pos = self.val_into_pos(val), 0
                        setattr(self, self.grabbed,
                                self.pos_into_val_grey(touch.x))
                else:
                    setattr(self, self.grabbed,
                            self.pos_into_val_grey(touch.x))
                self.label_val = getattr(self, self.grabbed)
            return True

    # Used by: called if touch is going up
    def on_touch_up(self, touch):
        if touch.grab_current == self:
            if self.grabbed == "slider":
                self.value_pos = touch.pos
                self.label_val = self.value
            touch.ungrab(self)
            return True

    def on_size(self, slider, size):
        new_size = (int(max(self.size[0] / 100, 8)), int(self.size[1] / 1.5))
        self.bot_restr_canvas.size = new_size
        self.top_restr_canvas.size = new_size
        self.bot_restr.size = new_size
        self.top_restr.size = new_size


    # Increase the value and schedules one increase every .1 s.
    # Used by: called by pressed button "up"
    def start_up(self, obj):
        idx = min((abs(self.data - self.value)).argmin() + 1,
                    self.data.size - 1)
        val = float(self.data[idx])
        self.value = min(self.top_val, val)
        self.label_val = self.value
        Clock.schedule_interval(self.change_up, .5)

    # Increase the value once in a given time.
    # Used by: start_up, PLayout.change_two_sliders
    #          PLayout.up_with_keys
    def change_up(self, dt):
        idx = min((abs(self.data - self.value)).argmin() + 1,
                    self.data.size - 1)
        val = float(self.data[idx])
        self.value = min(self.top_val, val)
        self.label_val = self.value

    # Interrupt the scheduled increase of the value.
    # Used by: called by released button "up"
    def stop_up(self, obj):
        Clock.unschedule(self.change_up)

    # Decrease the value and schedule one decrease every .1 s.
    # Used by: called by pressed button "down"
    def start_down(self, obj):
        idx = max((abs(self.data - self.value)).argmin() - 1, 0)
        val = float(self.data[idx])
        self.value = max(self.bot_val, val)
        self.label_val = self.value
        Clock.schedule_interval(self.change_down, .5)

    # Decrease value once in a given time.
    # Used by: start_down, PLayout.change_two_sliders,
    #          PLayout.down_with_keys
    def change_down(self, dt):
        idx = max((abs(self.data - self.value)).argmin() - 1, 0)
        val = float(self.data[idx])
        self.value = max(self.bot_val, val)
        self.label_val = self.value

    # Interrupt the scheduled decrease of the value.
    # Used by: called by released button "down"
    def stop_down(self, obj):
        Clock.unschedule(self.change_down)

    def get_data_idx(self):
        idx = np.argmin(abs(self.data - self.value))
        return idx

    def get_grey_idx(self):
        idx = np.argmin(abs(self.grey - self.value))
        return idx

    # Used by: PLayout.restrictor_focus
    def unfocus_restrictor(self, d):
        self.grabbed = "slider"
        self.label_val = self.value


# History class
class History:

    # Used by: PLayout.__init__
    def __init__(self, size):
        self.size = size
        self.curr_idx = size - 1
        self.hist = -1 * np.ones(size, dtype=np.int64)
        self.pointer = size - 1
        self.down_border = (self.curr_idx + 1) % self.size

    # Used by: PLayout.highlight_datapoint
    def show(self):
        return int(self.hist[self.pointer])

    # Overwrite the next array entry in order to save a new state.
    # Used by: PLayout.on_val
    def save(self, value):
        if self.pointer == self.curr_idx:
            if self.down_border == (self.pointer + 1) % self.size:
                self.down_border = (self.pointer + 2) % self.size
        self.curr_idx = (self.pointer + 1) % self.size
        self.pointer = self.curr_idx
        self.hist[self.curr_idx] = value

    # Change pointer downwards/upwards to get an older/newer state.
    # Used by: PLayout.undo, PLayout.redo
    def change(self, kw):
        if kw == "up":
            if self.pointer != self.curr_idx:
                self.pointer = (self.pointer+1) % self.size
                return int(self.hist[self.pointer])
        if kw == "down":
            if self.hist[(self.pointer-1) % self.size] == -1:
                return -1
            if self.pointer != self.down_border:
                self.pointer = (self.pointer-1) % self.size
                return int(self.hist[self.pointer])
        return -1

    # Used by: PLayout.on_val
    def prnt(self):
        print(str(self.hist))


# Root widget
class PLayout(BoxLayout):
    unrestr = ObjectProperty(np.array([]), force_dispatch=True)
    grey = ObjectProperty(np.array([]), force_dispatch=True)
    line = None

    # Read data from csv-file and write it into five arrays.
    # Initialize the sliders and the history.
    # Plot f against third param and connect mpl events to kivy.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.names = ["volume_A_tilde", "conc_e", "temperature", "reaction_time",  "impurity_conc [min]", "prod_cost [min]", "mat_cost [min]"]
        #data_dump = np.load('../main/pareto_data.npz')
        data_dump = np.load('pareto_data.npz')
        self.N = len(self.names)
        self.Ne = data_dump[self.names[0]].shape[0]
        i = self.N
        for j in range(self.N):
            if "[min]" in self.names[j]:
                i = i - 1
                self.names[j], self.names[i] = self.names[i], self.names[j]
        self.Np = i
        self.data = np.zeros((self.N, self.Ne), dtype=np.float)
        self.data[0, :] = data_dump['volume_A_tilde']
        self.data[1, :] = data_dump['conc_e']
        self.data[2, :] = data_dump['temperature']
        self.data[3, :] = data_dump['reaction_time']
        self.data[4, :] = data_dump['impurity_conc']
        self.data[5, :] = data_dump['prod_cost']
        self.data[6, :] = data_dump['mat_cost']

        # idxarray = np.array(range(self.Ne - 1, -1, -1))
        # self.data[self.N - 1, :] = self.data[self.N - 2, idxarray]
        # self.names.append("f2 [min]")
        # self.names = [str("param" + str(i + 1)) for i in range(self.N)]
        self.history = History(7)
        self.unbind(grey=self.on_grey)
        self.init_kv()
        self.change_unrestr()
        for i in range(self.N):
            s = self.sliders[i]
            p = self.data[i]
            s.grey = np.unique(p)
            s.data = s.grey
        valuearray = np.empty(self.N, dtype=float)
        for i in range(self.N):
            valuearray[i] = self.sliders[i].value
        # idx = c_fun.search_one(valuearray, self.data, self.unrestr)
        idx = 0
        if idx == -1:
            idx = self.grey[0]
            self.history.save(idx)
            valuearray = np.array([self.data[i, idx] for i in range(self.N)])
            self.change_sliders_vals(valuearray, np.array(range(self.N)), highlight=False)
        else:
            self.history.save(idx)
        xh = self.data[self.x_idx, self.grey]
        yh = self.data[self.y_idx, self.grey]

        for plot in self.graph.plots:
            self.graph.remove_plot(plot)
        dx = self.data[self.x_idx, self.grey].max() - self.data[self.x_idx, self.grey].min()
        if dx == 0:
            dx = 20
        mn = self.data[self.x_idx, self.grey].min() - .1*dx
        mx = self.data[self.x_idx, self.grey].max() + .1*dx
        dy = self.data[self.y_idx, self.grey].max() - self.data[self.y_idx, self.grey].min()
        if dy == 0:
            dy = 40
        fmn = self.data[self.y_idx, self.grey].min() - .2*dy
        fmx = self.data[self.y_idx, self.grey].max() + .2*dy
        self.graph.xmin = float(mn)
        self.graph.xmax = float(mx)
        self.graph.ymin = float(fmn)
        self.graph.ymax = float(fmx)
        self.graph.x_ticks_major = (mx - mn) / 10
        self.graph.y_ticks_major = (fmx - fmn) / 5
        self.plot_grey = DotPlot(color=[1, 1, 1, 1])
        self.plot_grey.points = [(self.data[self.x_idx, i], self.data[self.y_idx, i]) for i in self.grey]
        self.selected = DotPlot(color=[.9, 1, 0, .8])
        self.selected.points = [(round(self.data[self.x_idx, idx], 6), round(self.data[self.y_idx, idx], 6))]
        for plot in self.graph.plots:
            self.graph.remove_plot(plot)
        self.graph.add_plot(self.plot_grey)
        self.graph.add_plot(self.selected)

        '''
        np.savez('pareto_data.npz', volume_A_tilde=self.data[0], conc_e=self.data[1], temperature=self.data[2],
                 reaction_time=self.data[3], impurity_conc=self.data[4], prod_cost=self.data[5], mat_cost=self.data[6])
        '''


        '''
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("plot", size="x-large")
        self.text = self.ax.text(0.01, 0.99, "selected: none",
                                 transform=self.ax.transAxes,
                                 va="top", size="large")
        self.ax.set_xlabel(self.names[self.x_idx], labelpad=-13, x=1.05, size="large")
        self.ax.set_ylabel(self.names[self.y_idx], size="large")
        #self.line_g, = self.ax.plot(xh, yh, "o", c=(.7,.7,.7))
        self.line, = self.ax.plot(self.data[self.x_idx, self.grey],
                                    self.data[self.y_idx, self.grey],
                                    "o", picker=6)
        self.selected, = self.ax.plot(round(self.data[self.x_idx, idx], 6),
                                      round(self.data[self.y_idx, idx], 6),
                                      "o", ms=12, alpha=0.4,
                                      color="yellow", visible=True)
        self.set_ax_lims()
        self.canv = self.fig.canvas
        self.graph.add_widget(self.canv)
        '''

        self._keyboard = Window.request_keyboard(self.close_keyboard, self)
        #self._keyboard.bind(on_key_down=self.on_press)
        #self.canv.mpl_connect("pick_event", self.on_pick)
        self.spinnerx.bind(text=self.update_x)
        self.spinnery.bind(text=self.update_y)
        self.bind(grey=self.on_grey)
        self.graph.bind(on_touch_down=self.on_pick)


    def on_grey(self, obj, value):
        self.graph.remove_plot(self.plot_grey)
        self.plot_grey = DotPlot(color=[1, 1, 1, 1])
        self.plot_grey.points = [(self.data[self.x_idx, i], self.data[self.y_idx, i]) for i in self.grey]
        self.graph.add_plot(self.plot_grey)
        dx = self.data[self.x_idx, self.grey].max() - self.data[self.x_idx, self.grey].min()
        if dx == 0:
            dx = 20
        mn = self.data[self.x_idx, self.grey].min() - .1*dx
        mx = self.data[self.x_idx, self.grey].max() + .1*dx
        dy = self.data[self.y_idx, self.grey].max() - self.data[self.y_idx, self.grey].min()
        if dy == 0:
            dy = 40
        fmn = self.data[self.y_idx, self.grey].min() - .2*dy
        fmx = self.data[self.y_idx, self.grey].max() + .2*dy
        self.graph.xmin = float(mn)
        self.graph.xmax = float(mx)
        self.graph.ymin = float(fmn)
        self.graph.ymax = float(fmx)
        self.graph.x_ticks_major = (mx - mn) / 10
        self.graph.y_ticks_major = (fmx - fmn) / 5
        '''
        self.line.set_data(self.data[self.x_idx, self.grey],
                             self.data[self.y_idx, self.grey])
        self.set_ax_lims()
        self.canv.draw()
        '''

    def on_unrestr(self, obj, value):
        a = fuse_arrays(self.data[self.x_idx, self.unrestr],
                              self.data[self.y_idx, self.unrestr])
        ac = np.ascontiguousarray(a).view(
            np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, idx = np.unique(ac, return_index=True)
        idx = np.unique(idx)
        self.grey = np.array(self.unrestr[idx])
        self.highlight_datapoint()

    def change_unrestr(self):
        #a = [[1, self.data[x, :], 1] for x in range(self.N)]
        b = np.zeros(self.N)
        t = np.zeros(self.N)
        i = 0
        for i in range(self.N):
            s = self.sliders[i]
            b[i] = s.bot_val
            t[i] = s.top_val
        d = np.array([((b[i] <= self.data[i]) & (self.data[i] <= t[i])) for i in range(self.N)])
        _, g = np.where(d)
        u, c = np.unique(g, return_counts=True)
        self.unrestr = u[np.where(c == self.N)]
        #diese methode ueberpruefen!
        #self.unrestr, = np.where(((a[x][0] <= a[x][1]) & (a[x][1] <= a[x][2]))
        #                         for x in range(len(a)))

    def change_all_sliders_data(self):
        for i in range(self.Np):
            valuearray = np.empty(self.N - 1, dtype=float)
            idxarray = np.empty(self.N -1, dtype=int)
            l = 0
            for j in range(self.Np):
                if j != i:
                    valuearray[l] = self.sliders[j].value
                    idxarray[l] = j
                    l += 1
            valuearray = valuearray[:l]
            idxarray = idxarray[:l]
            ind = search_all(valuearray, self.data[idxarray], self.unrestr)
            self.sliders[i].data = self.data[i, ind]
        for i in range(self.Np, self.N, 1):
            self.sliders[i].data = self.data[i, self.unrestr]

    def update_slider_pos(self, obj, size):
        w, h = size
        for i in range(self.N):
            self.sliders[i].pos = (.2*w, (4-i)*0.5/(self.N + 1)*h)

    def label_val_callback(self, slider, val):
        i, = np.where(slider == self.sliders)
        i = i[0]
        if self.sliders[i].label_val > 1e-9:
            exponent = np.floor(np.log10(np.abs(self.sliders[i].label_val))).astype(int)
            if (exponent >= 6) or (exponent <= -6):
                text = '{:.2e}'.format(self.sliders[i].label_val)
            elif exponent > 0:
                precision = 5 - exponent
                text = '{:.{}f}'.format(self.sliders[i].label_val, precision)
            else:
                precision = 5
                text = '{:.5f}'.format(self.sliders[i].label_val)
        else:
            precision = 5
            text = '{:.5f}'.format(self.sliders[i].label_val)
        self.lbl2[i].text = text

    def init_kv(self):
        self.x_idx = 3
        self.y_idx = 4
        self.spacing = 10
        self.padding = (5, 5)
        self.orientation = "vertical"
        stklayout = StackLayout(orientation="lr-tb",
                             spacing=5,
                             padding=10,
                             size_hint=(1, .55)
                            )
        self.graph_layout = BoxLayout(orientation="vertical",
                                      size_hint=(1, .8)
                                     )
        self.graph = MyGraph()
        self.graph.background_color = [0, 0, 0, 0]
        self.graph.tick_color = [1, 1, 1, 1]
        self.graph.xlabel = self.names[self.x_idx]
        self.graph.ylabel = self.names[self.y_idx]
        self.graph.x_grid_label = True
        self.graph.y_grid_label = True
        self.graph_layout.add_widget(self.graph)
        lbl1 = Label(text="plot",
                     size_hint=(.1, .1)
                    )
        lbl2 = Label(text="against",
                     size_hint=(.15, .1)
                    )
        lbl3 = Label(size_hint=(.05, .1))
        self.spinnerx = Spinner(text=self.names[self.x_idx],
                                values=self.names[:self.y_idx] + self.names[self.y_idx + 1:],
                                size_hint=(.25, .1)
                               )
        self.spinnery = Spinner(text=self.names[self.y_idx],
                                values=self.names[:self.x_idx] + self.names[self.x_idx + 1:],
                                size_hint=(.25, .1)
                               )
        btn_undo = Button(text="undo",
                          size_hint=(.1, .1),
                          on_press=self.undo
                         )
        btn_redo = Button(text="redo",
                          size_hint=(.1, .1),
                          on_press=self.redo
                         )
        stklayout.add_widget(lbl1)
        stklayout.add_widget(self.spinnery)
        stklayout.add_widget(lbl2)
        stklayout.add_widget(self.spinnerx)
        stklayout.add_widget(lbl3)
        stklayout.add_widget(btn_undo)
        stklayout.add_widget(btn_redo)
        stklayout.add_widget(self.graph_layout)
        self.add_widget(stklayout)

        pmin = np.empty(self.N, dtype=float)
        pmax = np.empty(self.N, dtype=float)
        for i in range(self.N):
            pmin[i] = self.data[i].min()
            pmax[i] = self.data[i].max()
        self.sliders = np.empty(self.N, dtype=np.dtype(object))
        rel_padding = (self.padding[1] + self.padding[3]) / self.height / self.N
        rel_spacing = self.spacing * (self.N + 1) / self.N / self.height
        self.rel_widget_height = 0.55 / self.N - rel_spacing - rel_padding
        #widget_height = self.rel_widget_height*self.height
        for i in range(self.N):
            self.sliders[i] = MySlider(max=float(pmax[i]),
                                      min=float(pmin[i]),
                                      orientation="horizontal",
                                      size_hint=(.7, 1)
                                     )
            self.sliders[i].bind(on_touch_up=self.restrictor_focus)
            self.sliders[i].bind(value=self.on_val)
            self.sliders[i].bind(bot_val=self.on_restr)
            self.sliders[i].bind(top_val=self.on_restr)
            self.sliders[i].bind(size=self.update_slider_pos)
            self.sliders[i].bind(label_val=self.label_val_callback)
        self.lbl2 = np.zeros(self.N, dtype=np.dtype(object))
        self.leftbox = BoxLayout(orientation="vertical",
                                       padding=(0, 0),
                                       spacing=10,
                                       size_hint=(.05, 1)
                                      )
        self.rightbox = BoxLayout(orientation="vertical",
                                       padding=(0, 0),
                                       spacing=10,
                                       size_hint=(.95, 1)
                                      )
        self.lowbox = BoxLayout(orientation="horizontal",
                                       padding=(0, 0),
                                       spacing=10,
                                       size_hint=(1, .45)
                                      )
        self.add_widget(self.lowbox)
        self.lowbox.add_widget(self.leftbox)
        self.lowbox.add_widget(self.rightbox)
        self.boxlayouts = np.zeros(self.N, dtype=np.dtype(object))
        for i in range(self.N):
            self.boxlayouts[i] = BoxLayout(orientation="horizontal",
                                           padding=(0, 0),
                                           spacing=10,
                                           size_hint=(1, self.rel_widget_height)
                                          )
            lbl1 = Label(text=str(self.names[i]),
                         size_hint=(.15, 1),
                         valign="middle",
                         halign="left"
                        )
            if self.sliders[i].label_val > 1e-9:
                exponent = np.floor(np.log10(np.abs(self.sliders[i].label_val))).astype(int)
                if (exponent >= 6) or (exponent <= -6):
                    lbl2 = Label(text=('{:.2e}'.format(self.sliders[i].label_val)),
                                 size_hint=(.05, 1),
                                 valign="middle",
                                 halign="left"
                                )
                elif exponent > 0:
                    precision = 5 - exponent
                    lbl2 = Label(text=('{:.{}f}'.format(self.sliders[i].label_val, precision)),
                                 size_hint=(.05, 1),
                                 valign="middle",
                                 halign="left"
                                )
                else:
                    precision = 5
                    lbl2 = Label(text=('{:.5f}'.format(self.sliders[i].label_val)),
                                 size_hint=(.05, 1),
                                 valign="middle",
                                 halign="left"
                                )
            else:
                precision = 5
                lbl2 = Label(text=('{:.5f}'.format(self.sliders[i].label_val)),
                             size_hint=(.05, 1),
                             valign="middle",
                             halign="left"
                            )
            layout = BoxLayout(orientation="vertical",
                               size_hint=(.1, 1)
                               )
            btn_up = Button(text="+",
                            on_press=self.sliders[i].start_up,
                            on_release=self.sliders[i].stop_up
                           )
            btn_down = Button(text="-",
                              on_press=self.sliders[i].start_down,
                              on_release=self.sliders[i].stop_down
                             )
            layout.add_widget(btn_up)
            layout.add_widget(btn_down)
            self.boxlayouts[i].add_widget(lbl1)
            self.boxlayouts[i].add_widget(lbl2)
            self.boxlayouts[i].add_widget(self.sliders[i])
            self.boxlayouts[i].add_widget(layout)
            self.rightbox.add_widget(self.boxlayouts[i])
            self.lbl2[i] = lbl2
        lbds = Label(text="Objective\nspace",
                     size_hint=(1, float((self.Np - 1) / self.N)),
                     valign="middle",
                     halign="center"
                    )
        lbos = Label(text="Design\nspace",
                     size_hint=(1, float(1 - (self.Np - 1) / self.N)),
                     valign="middle",
                     halign="center"
                    )
        self.leftbox.add_widget(lbos)
        self.leftbox.add_widget(lbds)
        self.bind(size=self.on_size)

    def on_size(self, obj, size):
        rel_padding = (self.padding[1] + self.padding[3]) / self.height / self.N
        rel_spacing = self.spacing * (self.N + 1) / self.N / self.height
        self.rel_widget_height = 0.55 / self.N - rel_spacing - rel_padding
        for bl in self.boxlayouts:
            bl.size_hint = (1, self.rel_widget_height)

    # Initialize the 5 sliders with their params.
    # Set sliderx, slidery, paramx and paramy and bind them to their
    # callbacks.
    # Used by: nothing atm
    def init_sliders(self):
        pmin = np.empty(self.N, dtype=float)
        pmax = np.empty(self.N, dtype=float)
        for i in range(self.N):
            pmin[i] = self.data[i].min()
            pmax[i] = self.data[i].max()
        self.init_kv(pmin, pmax)
        for i in range(self.N):
            s = self.sliders[i]
            p = self.data[i]
            s.max = float(p.max())
            s.min = float(p.min())
            s.bind(on_touch_up=self.restrictor_focus)
            s.bind(value=self.on_val)
            s.bind(bot_val=self.on_restr)
            s.bind(top_val=self.on_restr)

    # Use search_one to get index of current state.
    # Use get_new_value if there is no such index.
    # Save the state in history.
    # Highlight current datapoint and plot the changes.
    # Used by: called if the value of a slider changes
    def on_val(self, obj, val):
        obj_idx, = np.where(obj == self.sliders)
        obj_idx = int(obj_idx)
        idx = search_all(np.array([float(val)]), self.data[obj_idx:, :], self.unrestr)
        if len(idx) == 1:
            idx = idx[0]
            if idx == -1:
                return
        elif len(idx) > 1:
            idx = np.array(idx)
            A = self.data[:self.Np, :]
            B = np.array([self.sliders[i].value for i in range(self.N)])
            B[obj_idx] = val
            distances = np.empty(len(idx))
            j = 0
            for i in idx:
                distances[j] = np.sqrt(np.sum((A[:, i] - B[:self.Np])**2))
                j = j + 1
            idx = idx[np.argmin(distances)]
        else:
            return
        self.history.save(idx)
        self.change_sliders_vals(self.data[:, idx], np.array(range(self.N)))
        return

    # Get a near datapoint.
    # Set value of sliders to the value of this datapoint.
    # Used by: called by a click on mpl content
    def on_pick(self, graph, event):
        indmin = self.graph.touched(event.pos[0], event.pos[1])
        if indmin:
            valx = float(self.data[self.x_idx, self.grey[indmin]])
            valy = float(self.data[self.y_idx, self.grey[indmin]])
            if valy == self.sliders[self.y_idx].value:
                sliderx_pos = self.sliders[self.x_idx].val_into_pos(valx)
                self.sliders[self.x_idx].value_pos = sliderx_pos, 0
            else:
                par = (self.sliders[self.x_idx], valx, self.sliders[self.y_idx], valy)
                self.change_two_sliders(*par)


    # Unbind a slider to change his value without a callback.
    # Change value to a given value or to the next higher/lower value.
    # Used by: on_pick, browse_with_keys, get_new_value
    def change_two_sliders(self, obj1, val1, obj2, val2, val=True):
        obj1_idx, = np.where(obj1 == self.sliders)
        obj1_idx = int(obj1_idx)
        obj2_idx, = np.where(obj2 == self.sliders)
        obj2_idx = int(obj2_idx)
        valuearray = np.array([val1, val2])
        idxarray = np.array([obj1_idx, obj2_idx])
        idx = search_all(valuearray, self.data[idxarray, :], self.unrestr)
        if len(idx) == 1:
            idx = idx[0]
            if idx == -1:
                return
        elif len(idx) > 1:
            idx = np.array(idx)
            A = self.data[:self.Np, :]
            B = np.array([self.sliders[i].value for i in range(self.N)])
            B[obj1_idx] = val1
            B[obj2_idx] = val2
            distances = np.empty(len(idx))
            j = 0
            for i in idx:
                distances[j] = np.sqrt(np.sum((A[:, i] - B[:self.Np])**2))
                j = j + 1
            idx = idx[np.argmin(distances)]
        else:
            return
        self.history.save(idx)
        valuearray = self.data[:, idx]
        self.change_sliders_vals(valuearray, np.array(range(self.N)))

    # Browse through plot or undo/redo by using the keyboard.
    # Used by: called if a key is pressed
    def on_press(self, keyboard, keycode, text, modifiers):
        if keycode[1] not in ("up", "down", "left", "right", "n", "p"):
            return
        sliderx = self.sliders[self.x_idx]
        slidery = self.sliders[self.y_idx]
        if keycode[1] == "right":
            self.up_with_keys(sliderx, slidery)
        elif keycode[1] == "up":
            self.up_with_keys(slidery, sliderx)
        elif keycode[1] == "left":
            self.down_with_keys(sliderx, slidery)
        elif keycode[1] == "down":
            self.down_with_keys(slidery, sliderx)
        elif keycode[1] == "n":
            self.redo(keyboard)
        else:
            self.undo(keyboard)

    # Browse upwards with keys.
    # Used by: on_press
    def up_with_keys(self, obj1, obj2):
        if not obj1:
            return
        if obj1.get_data_idx() >= obj1.data.size - 1:
            if obj2:
                obj2_grey_idx = obj2.get_grey_idx()
                val2 = obj2.grey[obj2_grey_idx + 1]
                if obj2.grey[obj2_grey_idx] < obj2.top_val:
                    obj1_idx, = np.where(obj1 == self.sliders)
                    obj1_idx = int(obj1_idx)
                    obj2_idx, = np.where(obj2 == self.sliders)
                    obj2_idx = int(obj2_idx)
                    valuearray = np.array([val2])
                    idxarray = np.array([obj2_idx])
                    idx = search_all(valuearray, self.data[idxarray, :], self.unrestr)
                    if len(idx) == 1:
                        idx = idx[0]
                        if idx == -1:
                            return
                    elif len(idx) > 1:
                        idx = np.array(idx)
                        val1 = min(self.data[obj1_idx, idx])
                        idx = idx[np.where(self.data[obj1_idx, idx] == val1)]
                        A = self.data[:self.Np, :]
                        B = np.array([self.sliders[i].value for i in range(self.N)])
                        B[obj1_idx] = val1
                        B[obj2_idx] = val2
                        distances = np.empty(len(idx))
                        j = 0
                        for i in idx:
                            distances[j] = np.sqrt(np.sum((A[:, i] - B[:self.Np])**2))
                            j = j + 1
                        idx = idx[np.argmin(distances)]
                    else:
                        return
                    self.history.save(idx)
                    valuearray = self.data[:, idx]
                    self.change_sliders_vals(valuearray, np.array(range(self.N)))
        else:
            obj1.change_up(.1)

    # Browse downwards with keys.
    # Used by: on_press
    def down_with_keys(self, obj1, obj2):
        if not obj1:
            return
        if obj1.get_data_idx() <= 0:
            if obj2:
                obj2_grey_idx = obj2.get_grey_idx()
                val2 = obj2.grey[obj2_grey_idx - 1]
                if obj2.grey[obj2_grey_idx] > obj2.bot_val:
                    obj1_idx, = np.where(obj1 == self.sliders)
                    obj1_idx = int(obj1_idx)
                    obj2_idx, = np.where(obj2 == self.sliders)
                    obj2_idx = int(obj2_idx)
                    valuearray = np.array([val2])
                    idxarray = np.array([obj2_idx])
                    idx = search_all(valuearray, self.data[idxarray, :], self.unrestr)
                    if len(idx) == 1:
                        idx = idx[0]
                        if idx == -1:
                            return
                    elif len(idx) > 1:
                        idx = np.array(idx)
                        val1 = max(self.data[obj1_idx, idx])
                        idx = idx[np.where(self.data[obj1_idx, idx] == val1)]
                        A = self.data[:self.Np, :]
                        B = np.array([self.sliders[i].value for i in range(self.N)])
                        B[obj1_idx] = val1
                        B[obj2_idx] = val2
                        distances = np.empty(len(idx))
                        j = 0
                        for i in idx:
                            distances[j] = np.sqrt(np.sum((A[:, i] - B[:self.Np])**2))
                            j = j + 1
                        idx = idx[np.argmin(distances)]
                    else:
                        return
                    self.history.save(idx)
                    valuearray = self.data[:, idx]
                    self.change_sliders_vals(valuearray, np.array(range(self.N)))
        else:
            obj1.change_down(.1)

    def change_first_slider_data(self, obj1, obj2, val2):
        valuearray = np.empty(self.N - 1, dtype=float)
        j = 0
        for i in range(self.Np):
            s = self.sliders[i]
            p = self.data[i]
            if s != obj1:
                if s == obj2:
                    valuearray[j] = val2
                    p2 = p
                else:
                    valuearray[j] = s.value
                j = j + 1
            else:
                idxarray = np.array([l for l in range(self.Np) if l != i])
                p1 = p
        valuearray = valuearray[:idxarray.shape[0]]
        ind = search_all(valuearray, self.data[idxarray], self.unrestr)
        obj1.data = p1[ind]

    # Call plot_params to plot with new value of restrictor.
    # Used by: called if the value of a restrictor changes
    def on_restr(self, restr, val):
        self.change_unrestr()

    # Update x axis by setting a new x sliparam and plotting it.
    # Used by: called if the value of spinnerx changes
    def update_x(self, spinner, text):
        self.x_idx, = np.where(text == np.array(self.names))
        self.x_idx = int(self.x_idx)
        self.graph.xlabel = self.names[self.x_idx]
        #self.ax.set_xlabel(self.names[self.x_idx])
        self.unbind(grey=self.on_grey)
        self.grey = np.array([1])
        self.bind(grey=self.on_grey)
        self.on_unrestr(None, 0)
        self.highlight_datapoint()
        self.spinnery.values = self.names[:self.x_idx] + self.names[self.x_idx + 1:]

    # Set a new pair of slider and param as paramx and sliderx.
    # Used by: nothing atm
    def set_slider_param_x(self, text):
        if text == "param1":
            self.x_idx = 0
            self.spinnery.values = ("o1", "o2", "o3", "param2",
                                    "param3", "param4")
        if text == "param2":
            self.x_idx = 1
            self.spinnery.values = ("o1", "o2", "o3", "param1",
                                    "param3", "param4")
        if text == "param3":
            self.x_idx = 2
            self.spinnery.values = ("o1", "o2", "o3", "param2",
                                    "param1", "param4")
        if text == "param4":
            self.x_idx = 3
            self.spinnery.values = ("o1", "o2", "o3", "param2",
                                    "param3", "param1")

    # Update y axis by setting a new y sliparam and plotting it.
    # Used by: called if the value of spinnery changes
    def update_y(self, spinner, text):
        self.y_idx, = np.where(text == np.array(self.names))
        self.y_idx = int(self.y_idx)
        #self.ax.set_ylabel(self.names[self.y_idx])
        self.graph.ylabel = self.names[self.y_idx]
        # self.set_slider_param_y(text)
        # where funktioniert evt nicht, ansonsten set slider param reanimieren
        self.unbind(grey=self.on_grey)
        self.grey = np.array([1])
        self.bind(grey=self.on_grey)
        self.on_unrestr(None, 0)
        self.highlight_datapoint()
        self.spinnerx.values = self.names[:self.y_idx] + self.names[self.y_idx + 1:]

    # Set a new pair of slider and param as paramy and slidery.
    # Used by: nothing atm
    def set_slider_param_y(self, text):
        if text == "param1":
            self.y_idx = 0
            self.spinnerx.values = ("param2", "param3", "param4")
        if text == "param2":
            self.y_idx = 1
            self.spinnerx.values = ("param1", "param3", "param4")
        if text == "param3":
            self.y_idx = 2
            self.spinnerx.values = ("param1", "param2", "param4")
        if text == "param4":
            self.y_idx = 3
            self.spinnerx.values = ("param1", "param2", "param3")

    # Unfocus restrictor after 5 seconds.
    # When a restrictor is focused the label shows his value.
    # Used by: called if a touch is going up on a restrictor
    def restrictor_focus(self, obj, touch):
        Clock.unschedule(obj.unfocus_restrictor)
        if obj.grabbed != "slider":
            Clock.schedule_once(obj.unfocus_restrictor, 5)

    # Used by: called if the keyboard disconnects
    def close_keyboard(self):
        try:
            self._keyboard = Window.request_keyboard(self.close_keyboard, self)
        except:
            self._keyboard.unbind(on_key_down=self.on_press)
            self._keyboard = None
            print("keyboard closed")

    # Use np.unique to find all unique tuples.
    # Return one x array and one y array with the unique values.
    # Used by: plot
    def filter(self, ar1, ar2):
        a = fuse_arrays(ar1, ar2)
        bot_restr = np.ascontiguousarray(a).view(
            np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, idx = np.unique(bot_restr, return_index=True)
        idx = np.unique(idx)
        unique = a[idx]
        xh = np.zeros(unique.size//2)
        yh = np.zeros(unique.size//2)
        for i in range(unique.size//2):
            xh[i] = unique[i][0]
            yh[i] = unique[i][1]
        return xh, yh

    # Draw a yellow circle around the datapoint set by the sliders
    # and plot its value.
    # Used by: on_val, plot_params
    def highlight_datapoint(self):
        pass
        if self.history.show() == -1:
            return
        y = self.data[self.y_idx, self.history.show()]
        x = self.data[self.x_idx, self.history.show()]
        self.graph.remove_plot(self.selected)
        self.selected = DotPlot(color=[.9, 1, 0, .8])
        self.selected.points = [(x, y)]
        self.graph.add_plot(self.selected)

        '''
        self.selected.set_data(x, y)
        self.text.set_text("selected: (%s, %s)" %(str(x), str(y)))
        self.selected.set_visible(True)
        self.canv.draw()
        '''

    def change_sliders_data_no_calls(self, value_array, idx_array):
        for i in idx_array:
            if i >= self.Np:
                self.sliders[i].data = self.data[i, self.unrestr]
            else:
                s = self.sliders[i]
                p = self.data[i]
                valuearray = np.empty(self.Np - 1, dtype=float)
                idxarray = np.empty(self.Np -1, dtype=int)
                l = 0
                m = 0
                for j in range(self.Np):
                    if j != i:
                        if j in idx_array:
                            valuearray[l] = value_array[m]
                            m = m + 1
                        else:
                            valuearray[l] = self.sliders[j].value
                        idxarray[l] = j
                        l += 1
                ind = search_all(valuearray, self.data[idxarray], self.unrestr)
                self.sliders[i].data = self.data[i, ind]
        """
        for i in range(self.N):
            s = self.sliders[i]
            p = self.data[i]
            slival = [1., self.data1, 1., self.data1, 1., self.data1,
                      self.unrestr]
            i = 0
            for ss,pp in self.sliparam:
                if ss != s:
                    if ss == self.sliders1:
                        slival[i] = v1
                    if ss == self.sliders2:
                        slival[i] = v2
                    if ss == self.sliders3:
                        slival[i] = v3
                    if ss == self.sliders4:
                        slival[i] = v4
                    slival[i+1] = pp
                    i = i + 2
            ind = c_fun.search_all_f(*slival)
            s.param = p[ind]
        """

    def change_sliders_vals(self, valuearray, idxarray, highlight=True):
        j = 0
        for i in idxarray:
            s = self.sliders[i]
            s.unbind(value=self.on_val)
            s.value_pos = float(s.val_into_pos(valuearray[j])), 0
            s.bind(value=self.on_val)
            j = j + 1
        if highlight:
            self.highlight_datapoint()


    # Get previous state from history object and restore it.
    # Change restrictors so the sliders are in an unrestricted area.
    # Used by: called if undo button is clicked
    def undo(self, obj):
        idx = self.history.change("down")
        if idx == -1:
            return
        for i in range(self.N):
            s = self.sliders[i]
            p = self.data[i]
            if s.bot_val > p[idx]:
                s.bot_val = float(p[idx])
            if s.top_val < p[idx]:
                s.top_val = float(p[idx])
        valuearray = np.array([self.data[i, idx] for i in range(self.N)])
        self.change_sliders_vals(valuearray, np.array(range(self.N)))
        self.change_unrestr()
        self.highlight_datapoint()

    # Get next state from history object and restore it.
    # Change restrictors so the sliders are in an unrestricted area.
    # Used by: called if redo button is clicked
    def redo(self, obj):
        idx = self.history.change("up")
        if idx == -1:
            return
        for i in range(self.N):
            s = self.sliders[i]
            p = self.data[i]
            if s.bot_val > p[idx]:
                s.bot_val = p[idx]
            if s.top_val < p[idx]:
                s.top_val = p[idx]
        valuearray = np.array([self.data[i, idx] for i in range(self.N)])
        self.change_sliders_vals(valuearray, np.array(range(self.N)))
        self.change_unrestr()
        self.highlight_datapoint()


# App
class ProjectApp(App):
    def build(self):
        p = PLayout()
        return p

if __name__ == "__main__":
    ProjectApp().run()
