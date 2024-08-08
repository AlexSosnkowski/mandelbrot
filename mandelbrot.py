#!/usr/bin/env python3

#this is a quick exploration of the mandelbrot set using python!
#By Alexander Sosnkowski 

#on a cuda environment try import cupy as np for speedup ?
import numpy as np
#import matplotlib.pyplot as plt
import tqdm
import click 
from PIL import Image as im 
#this gets rid of the welcome to pygame message 
import contextlib
with contextlib.redirect_stdout(None):
    import pygame


#an object containing all of out settings
class Settings:
    def __init__(self, x, y, max_iters, zoom, resolution, color_grad, out, screenshot_res):
        self.x = x
        self.y = y
        self.max_iters = max_iters
        self.zoom = zoom
        self.resolution = resolution
        self.color_grad = color_grad
        self.out = out
        self.screenshot_res = screenshot_res


#some code for generating color gradients 

def linear_gradient(start_rgb, end_rgb, num, final_rgb=[0,0,0]):
    colors = [start_rgb]
    m = [float(end_rgb[rgb] - start_rgb[rgb]) / num for rgb in range(3)] 
    
    for i in range(num - 1):
        #R G B
        out = []
        
        for j in range(3):
            out.append(
                int(start_rgb[j] + i * m[j])
                    )
        colors.append(out)            
    colors.append(final_rgb) #we want things in the mandelbrot set to be a distinct color 
    return np.array(colors)
    

#this is the most simple escape algorithm that checks if our point escapes our bounds ???? 
#check if it explodes when given as c iteratively for the function f(z) = z^2 + c starting at z = 0
def naive_escape(x, y, max_iter=1000, verbose=True, smooth=False):
        
    nx = np.shape(x)[0] ; ny = np.shape(x)[1] # we need to account for non square images also 
    done = np.ones([nx, ny]) 
    map = np.zeros([nx, ny])
    
    z = np.zeros([nx, ny]) 
    c = x + 1j * y 
    
    for i in tqdm.tqdm(range(max_iter), disable = not verbose):
        z = np.where(done == 1, np.square(z) + c, 0)  
        #z = np.square(z) + c this version is faster, so maybe reconsider alex 
        #done = np.where((done == 0) |  (np.square(np.real(z)) + np.square(np.imag(z)) > 4), 0, 1)
        done = np.where((done == 0) |  (np.absolute(z) > 2), 0, 1)
        map = map + done 
        
    #to get rid of the visual banding effect we do the following 
    #log base P where P is power of the mandelbrot equation 
    if smooth:
        map2 = map - np.log2(np.log(np.abs(z)) / np.log(max_iter)) 
        map = np.where(map2 != np.nan, map2, map)

    #now we want to normalize and exponentially map each points iteration count. 

    
    #normalize
    map = map / max_iter
    #map = map / np.max(map)
    return map  

def get_image(settings, screenshot=False, verbose=True):
    
    xmin, xmax = settings.x - (2 / settings.zoom), settings.x + (2 / settings.zoom) 
    ymin, ymax = settings.y - (2 / settings.zoom), settings.y + (2 / settings.zoom) 
    
    xdis = np.abs(xmax - xmin) ; ydis = np.abs(ymax - ymin) ; maxdis = max([xdis, ydis])
    xper = xdis / maxdis ; yper = ydis / maxdis #percentage of each axis, only relevant for square matricies 

    if screenshot:
        xaxis = np.linspace(xmin, xmax, int(settings.screenshot_res * xper))
        yaxis = np.linspace(ymin, ymax, int(settings.screenshot_res * yper))
    else:
        xaxis = np.linspace(xmin, xmax, int(settings.resolution * xper))
        yaxis = np.linspace(ymin, ymax, int(settings.resolution * yper))
    X, Y = np.meshgrid(xaxis, yaxis)
    #print(X) ; print("Spacer!!!!") ; print(Y)
    #later need to add the option for more advanced 
    mapping = naive_escape(X, Y, max_iter=settings.max_iters, verbose=verbose)   
    
    #finally we convert to an image, maybe make this a different function 
    
    

    """
    #old and slow method  
    image = np.repeat(mapping[:,:,np.newaxis], 3, axis=2)
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            
            #this version is for doing an exponential scale on the gradient, maybe better?
            #image[i, j, :] = color_grad[int(np.power(image[i,j,0], 0.5) * max_iters)]
            
            image[i, j, :] = settings.color_grad[int(image[i,j,0] * settings.max_iters)]
    """
    
    image = (mapping * settings.max_iters).astype('int32')
    #this is basically using a lookup table 
    image = settings.color_grad[image]
    image = image.astype('uint8')
    return np.flip(image, 0) # this is an alignment problem

#modified from https://www.pygame.org/pcr/transform_scale/
def aspect_scale(img, bx, by):
    """ Scales 'img' to fit into box bx/by.
     This method will retain the original image's aspect ratio """
    ix,iy = img.get_size()
    if ix > iy:
        # fit to width
        scale_factor = bx/float(ix)
        sy = scale_factor * iy
        if sy > by:
            scale_factor = by/float(iy)
            sx = scale_factor * ix
            sy = by
        else:
            sx = bx
    else:
        # fit to height
        scale_factor = by/float(iy)
        sx = scale_factor * ix
        if sx > bx:
            scale_factor = bx/float(ix)
            sx = bx
            sy = scale_factor * iy
        else:
            sy = by

    return pygame.transform.scale(img, (sx,sy))

# we may want to change this so that options are stored in a options object!!!!!!!!!!!!!
def realtime(settings, smooth=False):
    #get our initial mandelbrot set
    image = get_image(settings, verbose=False)
    image_t = np.transpose(image, (1, 0, 2))
    
    #setup pygame and other variables 
    pygame.init() 
    display = pygame.display.set_mode((np.shape(image_t)[0], np.shape(image_t)[1]), pygame.RESIZABLE) 
    #the transpose is needed because of how pygame represents images 
    surface = pygame.surfarray.make_surface(image_t)
    surface_s = pygame.transform.smoothscale(surface, display.get_size())
    
    title = "X: {}, Y: {}, Zoom: {}".format(settings.x, settings.y, settings.zoom)
    pygame.display.set_caption(title)
    
    #I should add a screenshot function 
    #elif is ugly here so maybe fix that
    #also 
    
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False ; break 
                elif event.key == pygame.K_LEFT:
                    settings.x -= 1/settings.zoom
                elif event.key == pygame.K_RIGHT:
                    settings.x += 1/settings.zoom
                elif event.key == pygame.K_UP:
                    settings.y += 1/settings.zoom
                elif event.key == pygame.K_DOWN:
                    settings.y -= 1/settings.zoom
                elif event.key == pygame.K_z:
                    settings.zoom *= 2
                elif event.key == pygame.K_x:
                    settings.zoom /= 2
                elif event.key == pygame.K_s:
                    settings.out = "{}.png".format(title)
                    screenshot(settings)
                    print("Screen shot taken and saved as file {}.png".format(title))
                image = get_image(settings, verbose=False)
                image_t = np.transpose(image, (1, 0, 2))
                surface = pygame.surfarray.make_surface(image_t)
                #surface_s = pygame.transform.smoothscale(surface, display.get_size())
                title = "X: {}, Y: {}, Zoom: {}".format(settings.x, settings.y, settings.zoom)
                pygame.display.set_caption(title)
                #print("X:", x, "Y:", y, "ZOOM: ", zoom) # make this into a text box 
                #its not great updating this every time a key is pressed 
        
        #surface_s = pygame.transform.smoothscale(surface, display.get_size())
        #set a black background 
        display.fill([0,0,0]);
        surface_s = aspect_scale(surface, display.get_size()[0], display.get_size()[1])
        rect = surface_s.get_rect();
        rect.center = display.get_rect().center
        display.blit(surface_s, rect)
        pygame.display.update()
    pygame.quit()
        
    

def screenshot(settings, smooth=False):
    
    image = get_image(settings, screenshot=True, verbose=True)


    data = im.fromarray(image, 'RGB') 
    
    if settings.out == '':
        data.show()
    else:
        data.save(settings.out) 

@click.command()
@click.option('--point', '-p', default="0, 0", help='The center of our image in format a, b where a is the real component and b is the imaginary (default origin)')
@click.option('--max_iters', '-i', '--iters',  default=500, help='Max iterations the escape algorithm tests')
@click.option('--resolution', '-r', default=200, help='How detailed to make the image')
@click.option('--zoom', '-z', default=1.0, help='How many times to zoom in')
@click.option('--color', '-c', default='sea', help='What color palet / gradient to use [x-ray, sun, seahorse]')
@click.option('--out', '-o', default='', help='Save to the specified file')
@click.option('--mode', '-m', default='realtime', help='Realtime or screenshot')
@click.option('--screenshot_res', '-sr', default=0, help='In realtime mode, the resolution of screenshots')
def mandel(point, max_iters, resolution, zoom, color, out, mode,screenshot_res):
    
    if screenshot_res == 0:
        screenshot_res = resolution
    
    match color: 
        case 'x-ray':
            color_grad = linear_gradient([204, 204, 255], [0, 0, 51], max_iters) 
        case 'sun':
            color_grad = linear_gradient([255,222,0], [224,51,27], max_iters, final_rgb=[255, 255, 255]) 
        case _:
            #this is the ocean or sea horse color scheme 
            color_grad = linear_gradient([70, 30, 180], [255, 128, 0], max_iters, final_rgb=[255,255,255]) 
    
    
    p = point.split(',') ; p = [float(xy) for xy in p]    
    x, y = p[0], p[1]
    
    settings = Settings(x, y, max_iters, zoom, resolution, color_grad, out, screenshot_res)
    
    match mode.lower():
        case 'realtime':
            realtime(settings)
        case _:
            screenshot(settings)
    



if __name__ == "__main__":
    mandel()



#things left to do

#see if I can't speed stuff up
#maybe some real time animations 
#finally, we may want to give the option to do different aspect ratios since the code can handel that
#and of course, julia sets 
