import tkinter as tk
import tkinter.ttk as ttk
import torch
import numpy as np
from cnn_model import MyCNN

class Paint(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.screen = [[0 for _ in range(28)] for _ in range(28)]
        self.model = torch.load('model_cnn.pt')
        self.model.eval()
        self.mx = 0
        self.mode = 0
        self.my = 0
        self.setUI()
        self.setGrid()

    def _rgbtohex(self, rgb):
        return "#%02x%02x%02x" % rgb  

    def draw(self, event):
        self.mx = event.x / self.scale
        self.my = event.y / self.scale
        self.setGrid()
        self.setPred()


    def b1Mouse(self, event):
        self.mode = 1
        self.draw(event)
    
    def b2Mouse(self, event):
        self.mode = 0
        self.draw(event)

    def clear(self):
        self.screen = [[0 for _ in range(28)] for _ in range(28)]
        self.setGrid()
        self.setPred()

    def setUI(self):
        self.parent.title("MNIST Paint")
        self.pack(fill=tk.BOTH, expand=1)
        self.parent.minsize(800, 520)
        self.parent.maxsize(800, 520)
        self.columnconfigure(4, weight=1)
        self.rowconfigure(11, weight=1)

        self.canv = tk.Canvas(self, bg="white", width=500, height=500)
        self.canv.grid(row=0, column=0, rowspan=11, padx=5, pady=5, sticky=tk.E + tk.W + tk.S + tk.N)
        self.canv.bind("<B1-Motion>", self.b1Mouse)
        self.canv.bind("<B2-Motion>", self.b2Mouse)

        self.num_lab = [tk.Label(self, text=str(i) + ':') for i in range(10)]
        self.num_prog_bar = [ttk.Progressbar(self, orient="horizontal", length=200, maximum=100) for i in range(10)]
        for i in range(10):
            self.num_lab[i].grid(row=i, column=1, padx=5)
            self.num_prog_bar[i].grid(row=i, column=2)
        
        self.clear_btn = tk.Button(self, text='Очитстить поле')
        self.clear_btn['command'] = self.clear
        self.clear_btn.grid(row=10, column=2)
        self.update()
        self.scale = self.canv.winfo_width() // 28.0



    def setGrid(self):
        x0 = 0
        y0 = 0
        self.canv.delete('all')
        for i in range(28):
            for j in range(28):
                dist = (j - self.mx) * (j - self.mx) + (i - self.my) * (i - self.my)
                if(dist < 1):
                    dist = 1
                dist *= dist
                if self.mode == 1:
                    self.screen[i][j] += 0.1 / dist
                elif self.mode == 0:
                    self.screen[i][j] -= 0.1 / dist
                if self.screen[i][j] > 1:
                     self.screen[i][j] = 1
                elif self.screen[i][j] < 0:
                     self.screen[i][j] = 0
                self.canv.create_rectangle(x0, y0, x0+self.scale, y0+self.scale, fill=self._rgbtohex((int(self.screen[i][j]*255), int(self.screen[i][j]*255), int(self.screen[i][j] * 255))))
                x0 = x0 + self.scale
            x0 = 0
            y0 = y0 + self.scale

    def setPred(self):
        tmp = np.array(self.screen) * 255
        tmp = torch.FloatTensor(tmp)
        tmp = tmp[None, None, :,:]
        self.pred = self.model(tmp)
        t = torch.nn.functional.softmax(self.pred, dim=1) * 100
        t = t.detach().numpy()[0]
        for i in range(10):
            self.num_prog_bar[i]['value']= t[i]
        

def main():
    global root
    root = tk.Tk()
    app = Paint(root)
    m = tk.Menu(root)
    root.config(menu=m)
    root.mainloop()

if __name__ == "__main__":
    main()
