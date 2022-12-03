import manim
from manim.utils.tex_templates import TexTemplateLibrary

class HelloLaTeX(manim.Scene):
    def construct(self):
        tex = manim.Tex(r'\LaTeX おはよう！', tex_template=TexTemplateLibrary.ctex, font_size=144)
        self.add(tex)
