import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.image import Texture
from kivy.graphics import Rectangle, Color
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

from camera import get_frame_raw
from collected_data import CollectedData

color_set = CollectedData("color_data", (720, 1280, 3))
depth_set = CollectedData("depth_data", (720, 1280))


class UI(BoxLayout):
    def __init__(self):
        super(UI, self).__init__()
        self.ids["check_0"].active = True

class GathererApp(App):
    def build(self):
        return UI()


class CameraHandler(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_data = None
        self.texture = None
        self.raw_data = (None, None)
        self.source = 0
        self.current_tag = 0
        with self.canvas:
            Rectangle(pos=self.pos, size=(self.size[0], self.size[1]), color=(0, 0, 0))
        Clock.schedule_once(self.start, .5)

    def start(self, _=None):
        Clock.schedule_interval(self.update, 1 / 60)

    def pause(self):
        Clock.unschedule(self.update)

    def set_source(self, source: 0 | 1):
        """set the source you want to display
        :param source: 0 for color, 1 for depth"""
        self.source = source

    def draw(self):
        if self.image_data is not None:
            with self.canvas:
                self.canvas.clear()
                Color(rgba=(1, 1, 1, 1))
                factor = max(self.texture.size[0] / self.size[0], self.texture.size[1] / self.size[1])
                Rectangle(pos=self.pos,
                          size=(self.texture.size[0] / factor, self.texture.size[1] / factor),
                          texture=self.texture)  # 在Canvas中绘制矩形，并绑定纹理

    # noinspection PyUnusedLocal
    def update(self, dt):
        self.raw_data = get_frame_raw()

        if self.source == 0:
            img = np.asanyarray(self.raw_data[0])
        else:
            # process the depth data to a gray scale picture
            img = np.asanyarray(self.raw_data[1])
            img = img.astype(np.float32)
            img *= (255 / 4000)
            img = 255 - img
            img = img.astype(np.uint8)
            img = np.dstack((img, img, img))

        img = np.flip(img, axis=0)
        self.image_data = img
        buf1 = img.tobytes()

        # noinspection PyArgumentList
        texture = Texture.create(size=(img.shape[1], img.shape[0]))  # 创建纹理对象
        texture.blit_buffer(buf1, colorfmt='bgr', bufferfmt='ubyte')  # 将数据绑定到纹理
        self.texture = texture

        self.draw()

    def set_current_tag(self, tag: 0 | 1):
        self.current_tag = tag

    def save_raw_data(self):
        depth = np.asanyarray(self.raw_data[1])
        color = np.asanyarray(self.raw_data[0])
        color_set.add_data(color, self.current_tag)
        depth_set.add_data(depth, self.current_tag)
        return

    def del_last_data(self):
        color_set.del_last_data(self.current_tag)
        depth_set.del_last_data(self.current_tag)


if __name__ == '__main__':
    app = GathererApp()

    app.run()
