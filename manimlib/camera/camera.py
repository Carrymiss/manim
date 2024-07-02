from __future__ import annotations

import moderngl
import numpy as np
import OpenGL.GL as gl
from PIL import Image

# 导入manim库的相关类和常量，用于相机的初始化和操作
from manimlib.camera.camera_frame import CameraFrame
from manimlib.constants import BLACK
from manimlib.constants import DEFAULT_FPS
from manimlib.constants import DEFAULT_PIXEL_HEIGHT, DEFAULT_PIXEL_WIDTH
from manimlib.constants import FRAME_WIDTH
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.mobject import Point
from manimlib.utils.color import color_to_rgba

from typing import TYPE_CHECKING

# 类型检查时的导入
if TYPE_CHECKING:
    from typing import Optional
    from manimlib.typing import ManimColor, Vect3
    from manimlib.window import Window


class Camera(object):
    """
    相机类，负责渲染场景和管理相机属性。

    参数:
    - window: 窗口对象，用于关联现代OpenGL上下文。
    - background_image: 背景图像的文件路径。
    - frame_config: 相机框架的配置字典。
    - pixel_width: 输出图像的像素宽度。
    - pixel_height: 输出图像的像素高度。
    - fps: 输出视频的帧率。
    - background_color: 背景颜色。
    - background_opacity: 背景透明度。
    - max_allowable_norm: 向量图形的最大允许规范。
    - image_mode: 图像模式，通常为RGBA。
    - n_channels: 图像通道数，通常为4（RGBA）。
    - pixel_array_dtype: 像素数组的数据类型。
    - light_source_position: 灯光源的位置。
    - samples: 抗锯齿采样数，用于多采样抗锯齿。
    """

    def __init__(
            self,
            window: Optional[Window] = None,
            background_image: Optional[str] = None,
            frame_config: dict = dict(),
            pixel_width: int = DEFAULT_PIXEL_WIDTH,
            pixel_height: int = DEFAULT_PIXEL_HEIGHT,
            fps: int = DEFAULT_FPS,
            background_color: ManimColor = BLACK,
            background_opacity: float = 1.0,
            max_allowable_norm: float = FRAME_WIDTH,
            image_mode: str = "RGBA",
            n_channels: int = 4,
            pixel_array_dtype: type = np.uint8,
            light_source_position: Vect3 = np.array([-10, 10, 10]),
            samples: int = 0,
    ):
        self.background_image = background_image
        self.window = window
        self.default_pixel_shape = (pixel_width, pixel_height)
        self.fps = fps
        self.max_allowable_norm = max_allowable_norm
        self.image_mode = image_mode
        self.n_channels = n_channels
        self.pixel_array_dtype = pixel_array_dtype
        self.light_source_position = light_source_position
        self.samples = samples

        # RGB值的最大值，基于像素数组的数据类型
        self.rgb_max_val: float = np.iinfo(self.pixel_array_dtype).max
        # 背景的RGBA值，用于初始化帧缓冲区
        self.background_rgba: list[float] = list(color_to_rgba(
            background_color, background_opacity
        ))
        self.uniforms = dict()  # 用于存储OpenGL着色器程序的uniform变量
        self.init_frame(**frame_config)
        self.init_context()
        self.init_fbo()
        self.init_light_source()

    def init_frame(self, **config) -> None:
        """
        初始化相机框架。

        参数:
        - **config: 框架的配置参数。
        """
        self.frame = CameraFrame(**config)

    def init_context(self) -> None:
        """
        初始化OpenGL上下文。

        如果没有关联的窗口，创建一个独立的上下文；否则使用窗口已有的上下文。
        """
        if self.window is None:
            self.ctx: moderngl.Context = moderngl.create_standalone_context()
        else:
            self.ctx: moderngl.Context = self.window.ctx

        # 启用一些OpenGL特性，如点大小程序和混合
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)

    def init_fbo(self) -> None:
        """
        初始化帧缓冲区对象（Frame Buffer Object，FBO）。

        创建用于渲染和存储图像的FBO，以及用于多采样抗锯齿的FBO。
        """
        # 用于最终渲染的多采样FBO
        # This is the buffer used when writing to a video/image file
        self.fbo_for_files = self.get_fbo(self.samples)

        # 用于绘制到的无采样FBO
        # This is the frame buffer we'll draw into when emitting frames
        self.draw_fbo = self.get_fbo(samples=0)

        if self.window is None:
            self.window_fbo = None
            self.fbo = self.fbo_for_files
        else:
            self.window_fbo = self.ctx.detect_framebuffer()
            self.fbo = self.window_fbo

        self.fbo.use()

    def init_light_source(self) -> None:
        """
        初始化灯光源。

        创建一个点对象表示灯光源的位置。
        """
        self.light_source = Point(self.light_source_position)

    def use_window_fbo(self, use: bool = True):
        """
        切换渲染目标到窗口FBO或多采样FBO。

        参数:
        - use: 布尔值，决定是否使用窗口FBO。
        """
        assert (self.window is not None)
        if use:
            self.fbo = self.window_fbo
        else:
            self.fbo = self.fbo_for_files

    def get_fbo(
            self,
            samples: int = 0
    ) -> moderngl.Framebuffer:
        """
        创建并返回一个新的帧缓冲区对象。

        参数:
        - samples: 用于多采样抗锯齿的样本数。

        返回:
        - 一个现代OpenGL帧缓冲区对象。
        """
        return self.ctx.framebuffer(
            color_attachments=self.ctx.texture(
                self.default_pixel_shape,
                components=self.n_channels,
                samples=samples,
            ),
            depth_attachment=self.ctx.depth_renderbuffer(
                self.default_pixel_shape,
                samples=samples
            )
        )

    def clear(self) -> None:
        """
        清空当前帧缓冲区。

        使用背景RGBA值进行清空。
        """
        self.fbo.clear(*self.background_rgba)

    def blit(self, src_fbo, dst_fbo):
        """
        使用Blit函数从一个帧缓冲区复制到另一个帧缓冲区。

        参数:
        - src_fbo: 源帧缓冲区。
        - dst_fbo: 目标帧缓冲区。
        """
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_fbo.glo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, dst_fbo.glo)
        gl.glBlitFramebuffer(
            *src_fbo.viewport,
            *dst_fbo.viewport,
            gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR
        )

    def get_raw_fbo_data(self, dtype: str = 'f1') -> bytes:
        """
        从绘制FBO中读取原始图像数据。

        参数:
        - dtype: 读取数据的类型。

        返回:
        - 一个字节流，包含FBO的图像数据。
        """
        self.blit(self.fbo, self.draw_fbo)
        return self.draw_fbo.read(
            viewport=self.draw_fbo.viewport,
            components=self.n_channels,
            dtype=dtype,
        )

    def get_image(self) -> Image.Image:
        """
        从当前帧缓冲区创建并返回一个PIL.Image对象。

        返回:
        - 一个PIL.Image对象，表示当前渲染的图像。
        """
        return Image.frombytes(
            'RGBA',
            self.get_pixel_shape(),
            self.get_raw_fbo_data(),
            'raw', 'RGBA', 0, -1
        )

    def get_pixel_array(self) -> np.ndarray:
        """
        获取当前帧缓冲区的像素数组。

        返回:
        - 一个NumPy数组，包含当前FBO的像素数据。
        """
        raw = self.get_raw_fbo_data(dtype='f4')
        flat_arr = np.frombuffer(raw, dtype='f4')
        arr = flat_arr.reshape([*reversed(self.draw_fbo.size), self.n_channels])
        arr = arr[::-1]
        # Convert from float
        return (self.rgb_max_val * arr).astype(self.pixel_array_dtype)

    def get_texture(self) -> moderngl.Texture:
        """
        从当前帧缓冲区创建并返回一个现代OpenGL纹理对象。

        返回:
        - 一个现代OpenGL纹理对象。
        """
        texture = self.ctx.texture(
            size=self.fbo.size,
            components=4,
            data=self.get_raw_fbo_data(),
            dtype='f4'
        )
        return texture

    def get_pixel_size(self) -> float:
        """
        获取单个像素的尺寸。

        返回:
        - 相机帧宽度与像素宽度的比值。
        """
        return self.frame.get_width() / self.get_pixel_shape()[0]

    def get_pixel_shape(self) -> tuple[int, int]:
        """
        获取当前帧缓冲区的像素形状（宽度，高度）。

        返回:
        - 像素形状的元组（宽度，高度）。
        """
        return self.fbo.size

    def get_pixel_width(self) -> int:
        """
        获取当前帧缓冲区的像素宽度。

        返回:
        - 像素宽度。
        """
        return self.get_pixel_shape()[0]

    def get_pixel_height(self) -> int:
        """
        获取当前帧缓冲区的像素高度。

        返回:
        - 像素高度。
        """
        return self.get_pixel_shape()[1]

    def get_aspect_ratio(self):
        """
        获取当前帧缓冲区的宽高比。

        返回:
        - 宽高比。
        """
        pw, ph = self.get_pixel_shape()
        return pw / ph

    def get_frame_height(self) -> float:
        """
        获取相机框架的高度。

        返回:
        - 相机框架的高度。
        """
        return self.frame.get_height()

    def get_frame_width(self) -> float:
        """
        获取相机框架的宽度。

        返回:
        - 相机框架的宽度。
        """
        return self.frame.get_width()

    def get_frame_shape(self) -> tuple[float, float]:
        """
        获取相机框架的形状（宽度，高度）。

        返回:
        - 相机框架形状的元组（宽度，高度）。
        """
        return (self.get_frame_width(), self.get_frame_height())

    def get_frame_center(self) -> np.ndarray:
        """
        获取相机框架的中心点位置。

        返回:
        - 相机框架中心点的三维向量。
        """
        return self.frame.get_center()

    def get_location(self) -> tuple[float, float, float]:
        """
        获取相机的位置。

        返回:
        - 相机位置的三维向量。
        """
        return self.frame.get_implied_camera_location()

    def resize_frame_shape(self, fixed_dimension: bool = False) -> None:
        """
        调整相机框架的尺寸以匹配像素的宽高比。

        参数:
        - fixed_dimension: 布尔值，确定在调整过程中是保持框架高度不变（True）还是宽度不变（False）。

        此方法会根据像素的宽高比自动调整相机框架的尺寸，可以选择固定框架的一个维度（高度或宽度），
        而让另一个维度按比例变化，以适应不同的显示需求。
        """
        frame_height = self.get_frame_height()
        frame_width = self.get_frame_width()
        aspect_ratio = self.get_aspect_ratio()

        # 根据fixed_dimension决定调整哪个维度
        if not fixed_dimension:
            frame_height = frame_width / aspect_ratio
        else:
            frame_width = aspect_ratio * frame_height

        # 应用新的尺寸到相机框架上
        self.frame.set_height(frame_height, stretch=True)
        self.frame.set_width(frame_width, stretch=True)

    # 渲染相关方法

    def capture(self, *mobjects: Mobject) -> None:
        """
        渲染一系列Mobject对象到当前帧缓冲区。

        参数:
        - *mobjects: 一系列要渲染的Mobject对象。

        此方法首先清除帧缓冲区，更新uniform变量，然后遍历每个Mobject对象进行渲染。
        如果相机与窗口关联且使用的不是窗口的默认帧缓冲区，还会将渲染结果复制到窗口的帧缓冲区。
        """
        self.clear()  # 清除当前帧缓冲区
        self.refresh_uniforms()  # 更新渲染所需的uniform变量
        self.fbo.use()  # 使用当前帧缓冲区开始渲染
        for mobject in mobjects:
            mobject.render(self.ctx, self.uniforms)  # 渲染每个Mobject
        # 如果有窗口且使用了独立的FBO，同步到窗口FBO
        if self.window is not None and self.fbo is not self.window_fbo:
            self.blit(self.fbo, self.window_fbo)

    def refresh_uniforms(self) -> None:
        """
        更新用于渲染的uniform变量。

        此方法收集相机框架的视图矩阵、光源位置和相机位置等信息，
        并将它们更新到uniform变量字典中，供渲染时使用。
        """
        frame = self.frame
        view_matrix = frame.get_view_matrix()  # 获取视图矩阵
        light_pos = self.light_source.get_location()  # 获取光源位置
        cam_pos = self.frame.get_implied_camera_location()  # 获取相机位置

        # 更新uniforms字典
        self.uniforms.update(
            view=tuple(view_matrix.T.flatten()),  # 视图矩阵转换并展平
            focal_distance=frame.get_focal_distance() / frame.get_scale(),  # 焦距除以缩放比例
            frame_scale=frame.get_scale(),  # 框架的缩放比例
            pixel_size=self.get_pixel_size(),  # 像素尺寸
            camera_position=tuple(cam_pos),  # 相机位置
            light_position=tuple(light_pos),  # 光源位置
        )


# 为了兼容旧场景而定义的类
class ThreeDCamera(Camera):
    """
    三维相机类，继承自Camera，主要是为了提供默认的多采样设置。

    该类主要目的是为了确保旧的场景脚本在升级后仍能正常工作，而不需要做大的修改。
    """

    def __init__(self, samples: int = 4, **kwargs):
        """
        初始化三维相机，预设了多采样抗锯齿的样本数。

        参数:
        - samples: 抗锯齿采样数，默认为4。
        - **kwargs: 其他关键字参数，传递给父类Camera的构造函数。
        """
        super().__init__(samples=samples, **kwargs)  # 调用父类构造函数，传入预设的多采样样本数
