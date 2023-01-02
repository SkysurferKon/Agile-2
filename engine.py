import os
import time
import math
import pygame
import random
import json
from functools import cache
from pygame.locals import *
from pygame import Surface
from pygame.time import set_timer
from pygame import Vector2
from pygame import Vector3
from pygame import display
from pygame.display import update
from pygame import Rect
from math import sin
from math import tan
from math import atan
from math import degrees
from math import sinh
from math import cosh
from math import tanh
from math import cos
from math import radians
from math import sqrt
from math import pi
from time import sleep
from random import randint
from pygame import image
from random import choice
from pygame.time import Clock
from pygame import gfxdraw
from pygame import mixer

os.environ["SDL_VIDEO_DRIVER"] = "directx"
pygame.init()
mixer.init()
'''if v_text2.a == 0 {
v_text2.rgb = vec4(
}
if (v_text2.x > 1.0 | | v_text2.x < 0.0 | | v_text2.y > 1.0 | | v_text2.y < 0.0){
f_color=vec4(0.0,0.0,0.0,1.0);
} else {
'''
# unused events
pygame.event.set_blocked(pygame.MOUSEMOTION)
pygame.event.set_blocked(pygame.K_RIGHTPAREN)
pygame.event.set_blocked(pygame.K_LEFTPAREN)
pygame.event.set_blocked(pygame.K_RIGHTBRACKET)
pygame.event.set_blocked(pygame.K_LEFTBRACKET)
pygame.event.set_blocked(pygame.K_MINUS)
pygame.event.set_blocked(pygame.K_BACKSLASH)
pygame.event.set_blocked(pygame.K_BACKSPACE)
pygame.event.set_blocked(pygame.K_BACKQUOTE)
pygame.event.set_blocked(pygame.K_QUOTE)
pygame.event.set_blocked(pygame.K_COLON)
pygame.event.set_blocked(pygame.K_COMMA)
pygame.event.set_blocked(pygame.K_DOLLAR)
pygame.event.set_blocked(pygame.K_UNDERSCORE)
pygame.event.set_blocked(pygame.MOUSEMOTION)
pygame.event.set_blocked(pygame.MOUSEWHEEL)


display.set_caption("Lightshifter")
#window = pygame.display.set_mode([1000,500],vsync=False, flags=OPENGL|DOUBLEBUF|SCALED|HWSURFACE|RESIZABLE|GL_ACCELERATED_VISUAL|HWACCEL|GL_DOUBLEBUFFER)
window = pygame.display.set_mode([500*2,250*2],vsync=False, flags=OPENGL|DOUBLEBUF|HWSURFACE|GL_ACCELERATED_VISUAL|HWACCEL|GL_DOUBLEBUFFER)
screen = Surface((1000,500)).convert((255, 65280, 16711680, 0))
gameupdate = USEREVENT + 0
mixer.set_num_channels(30)
set_timer(gameupdate,16)
clock = Clock()

black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

sndchannels = []
muschannels = []
spaceawaresndchannels = []
spaceawaremuschannels = []
scroll = [0,0]
truescroll = [0,0]
shockwaves = [[0,0,10000,100,1000] for i in range(5)]

screenshake = 0
screenshakeinterval = 0

@cache
def scale(image,size):
    image = pygame.transform.scale(image,size)
    return image
def Keys():
    return pygame.key.get_pressed()
def addvector(v1,v2):
    return[v1[0]+v2[0],v1[1]+v2[1]]
def subvector(v1,v2):
    return[v1[0]-v2[0],v1[1]-v2[1]]
def multiplyvector(v1,v2):
    return[v1[0]*v2[0],v1[1]*v2[1]]
def dividevector(v1,v2):
    return[v1[0]/v2[0],v1[1]/v2[1]]
def extract_color(img, color, add_surf=None):
    img = img.copy()
    img.set_colorkey(color)
    mask = pygame.mask.from_surface(img)
    surf = mask.to_surface(setcolor=(0, 0, 0, 0), unsetcolor=color)
    if add_surf:
        base_surf = pygame.Surface(img.get_size())
        base_surf.fill(color)
        add_surf = (add_surf[0].convert(), add_surf[1])
        add_surf[0].set_colorkey(add_surf[1])
        base_surf.blit(add_surf[0], (0, 0))
        base_surf.blit(surf, (0, 0))
        base_surf.set_colorkey((0, 0, 0))
        return base_surf
    else:
        return surf

class AnimatedFoliage:
    def __init__(self, image, color_chain, motion_scale=1):
        self.motion_scale = motion_scale
        self.base_image = image.copy()
        self.color_chain = color_chain
        self.layers = []

        for i, color in enumerate(color_chain[::-1]):
            if i == 0:
                self.layers.append(extract_color(self.base_image, color))
            else:
                self.layers.append(extract_color(self.base_image, color, add_surf=(self.layers[-1], color_chain[::-1][i - 1])))

        self.layers = self.layers[::-1]

    def find_leaf_point(self):
        while True:
            point = (int(random.random() * self.layers[0].get_width()), int(random.random() * self.layers[0].get_height()))
            color = self.layers[0].get_at(point)
            if list(color)[:3] != [0, 0, 0]:
                return point

    def render(self, surf, pos, m_clock=0, seed=14):
        surf.blit(pygame.transform.rotate(self.layers[0], math.sin(m_clock * 0.8 + (2.7 * seed)) * 1.2), (pos[0] + math.sin(m_clock * 1.7 + (2.7 * seed)) * 3 * self.motion_scale, pos[1] + math.sin(m_clock + (2.2 * seed)) * 2 * self.motion_scale))
        surf.blit(self.base_image, pos)
        for i, layer in enumerate(self.layers):
            if i != 0:
                surf.blit(pygame.transform.rotate(layer, math.sin(m_clock * 1.1) * 1.5), (pos[0] + math.sin(m_clock * (1.25 * i) + (2.7 * seed)) * 3 * self.motion_scale, pos[1] + math.sin(m_clock * (1.25 * i) + (2.2 * seed)) * 2 * self.motion_scale))
            else:
                surf.blit(layer, pos)


def getmovement(angle, speed):
    return cos(angle) * speed,sin(angle) * speed
def angleto(mp, tp):
    dx = tp[0] - mp[0]
    dy = tp[1] - mp[1]
    rads = math.atan2(-dy, dx)
    rads %= 2 * math.pi
    return math.degrees(rads)
def slowmotion(duration):
    lasttime = time.time()
    ptime = 0

    while 1:
        dt = time.time() - lasttime
        dt *= 60
        lasttime = time.time()
        for event in Events():
            if event.type == QUIT:
                exit()

        ptime += 1 * dt

        if ptime >= duration:
            break

        clock.tick(0)
def getvisiblefrompos(pos,extraspace=32):
    return -extraspace < pos[0] < 1000+extraspace and -extraspace < pos[1] < 500+extraspace
def getvisible(tile, extraspace=0):
    return tile.right + extraspace > 0 and tile.left - extraspace < 1000 and tile.bottom + extraspace > 0 and tile.top - extraspace < 500
def inproximity(target, inpos, range):
    if inpos[0] - range < target[0] < inpos[0] + range and inpos[1] - range < target[1] < inpos[1] + range:
        return True
    return False
def ispositive(number):
    if number == abs(number):
        return True
    else:
        return False
def isnegative(number):
    if number == abs(number):
        return False
    else:
        return True
def removescrollfromrect(r):
    global scroll

    r.x += scroll[0]
    r.y += scroll[1]

    return r
def addscroltorect(R):
    global scroll

    r = R.copy()
    r.x -= scroll[0]
    r.y -= scroll[1]

    return r
def removescroll(vector):
    global scroll

    vector = [vector[0] + scroll[0], vector[1] + scroll[1]]

    return vector
def addscroll(vector):
    global scroll

    vector = [vector[0] - scroll[0], vector[1] - scroll[1]]

    return vector
def drawaacircle(pos,radius,color=white,surface=screen):
    x,y = pos
    gfxdraw.aacircle(surface,x,y,radius,color)
    gfxdraw.filled_circle(surface,x,y,radius,color)
def colorswap(surf,old_c,new_c):
    img_copy = Surface(surf.get_size())
    img_copy.fill(new_c)
    surf.set_colorkey(old_c)
    img_copy.blit(surf,(0,0))
    return img_copy
def silhouette(surf, color=(255, 255, 255)):
    mask = pygame.mask.from_surface(surf)
    new_surf = colorswap(mask.to_surface(), (255, 255, 255), color)
    new_surf.set_colorkey((0, 0, 0))
    return new_surf

def outline(target, src, pos, color):
    s = silhouette(src, color=color)
    shifts = [[0, 1], [1, 0], [-1, 0], [0, -1]]
    for shift in shifts:
        target.blit(s, (pos[0] + shift[0]*2, pos[1] + shift[1]*2))

def rotate(image, angle):
    image = pygame.transform.rotate(image, angle)
    return image
def rotatecenter(image,angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image,angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image
def blitcenter(img,pos,special_flags=0,surf=screen):
    surf.blit(img,(round(pos[0] - img.get_width() / 2),round(pos[1] - img.get_height() / 2)),special_flags=special_flags)
def randfloat(a,b):
    data = []
    z = a
    for i in range(99999999):
        for x in range(10):
            data.append(z + x / 10)
        z += 1
        if z == b:
            break
    return choice(data)
def randfloatarr(a,b,division=10):
    data = []
    z = a

    while 1:
        for x in range(division):
            data.append(z + x / division)
        z += 1
        if z == b:
            break
    return data


pm = randfloatarr(-1,1)  # particle movement precalculation
sa = tuple([i for i in range(360)])  # spark angle precalculation
bm = randfloatarr(-2,2) #bit movement precalculation
bmy = randfloatarr(-4,2)


def Events():
    return pygame.event.get()
def fillalpha(img,alpha):
    img.set_alpha(alpha)#fill((255,255,255,int(alpha)),None,pygame.BLEND_RGBA_MULT)
    return img
def fadeout(img, fadespeed=5):
    screen = Surface((1000,500),flags=SRCALPHA)
    img = img.copy().convert_alpha()
    alpha = 255
    clock = Clock()

    while True:
        for event in Events():
            if event.type == pygame.QUIT:
                exit(0)
            if event.type == gameupdate:
                screen.fill(black)

                fillalpha(img,alpha)

                screen.blit(img.copy(),(0,0))
                alpha -= fadespeed
                window.blit(screen,(0,0))

        if alpha <= 50:
            break

        display.flip()
        clock.tick(0)

def fadein(img,fadespeed=10):
    global gameupdate
    src = img.copy().convert_alpha()
    alpha = 0
    clock = Clock()
    run = 1

    while run:
        for event in pygame.event.get():
            if event.type == QUIT:
                exit(0)
            if event.type == gameupdate:
                screen.fill(black)

                img = src.copy()
                img.fill((255,255,255,clamp(alpha,0,255)),None,pygame.BLEND_RGBA_MULT)

                screen.blit(img.copy(),(0,0))
                alpha += fadespeed

                if alpha >= 260:
                    screen.blit(img, (0,0))
                    run = 0

                screen.blit(screen,(0,0))
        display.update()
        clock.tick(0)
def colorize(img,newc):
    img = img.copy()
    img.fill((0,0,0,255),None,pygame.BLEND_RGBA_MULT)
    img.fill(newc[0:3] + (0,),None,pygame.BLEND_RGBA_ADD)

    return img
def loadmask(image,colorkey=None,scale=2):
    image = pygame.image.load("data/collision masks/" + image + ".png")
    image = pygame.transform.scale(image,(image.get_width() * scale,image.get_height() * scale))
    image.set_colorkey(colorkey)
    image.convert_alpha()
    mask = pygame.mask.from_surface(image)

    return mask
def loadsound(sound,vol=1):
    sound = pygame.mixer.Sound("data/sounds/" + sound)
    sound.set_volume(vol)
    return sound
def loadimage(image,colorkey=None,scale=2,convert=False):
    image = pygame.image.load("data/sprites/" + image + ".png")
    image = pygame.transform.scale(image,(image.get_width() * scale,image.get_height() * scale))
    image.set_colorkey(colorkey)
    if convert:image.convert()
    else:image.convert_alpha()

    return image
def loadanimation(filename,colorkey=None,scale=2, alpha=255):
    images = []

    path = os.listdir(f"data/sprites/{filename}")

    for i in range(len(path)):
        image = pygame.image.load(f"data/sprites/{filename}/{i}.png")
        image = pygame.transform.scale(image,(image.get_width() * scale,image.get_height() * scale))
        if alpha != 255:fillalpha(image,alpha)
        image.set_colorkey(colorkey)
        image.convert_alpha()
        images.append(image)

    return images
def playsnd(sound,loop=False,vol=None,sfxtype="snd"):
    global sndchannels,muschannels

    channel = pygame.mixer.find_channel(True)
    if vol is None: vol = sound.get_volume()

    if sfxtype == "snd":
        channel.set_volume(vol)
        channel.play(sound,loops=loop)
        sndchannels.append([channel,vol])

    if sfxtype == "mus":
        channel.set_volume(vol)
        channel.play(sound,loops=loop)
        muschannels.append([channel,vol])

    return channel
def playspaceawaresound(sound, pos, player, distanceheardfrom=0.003, loop=False, vol=None, sfxtype="snd",maxvol=None):
    global spaceawaresndchannels, spaceawaremuschannels

    channel = pygame.mixer.find_channel(True)
    if vol is None:
        vol = sound.get_volume()
        maxvol = sound.get_volume()

    if sfxtype == "snd":
        channel.set_volume(vol)
        channel.play(sound,loops=loop)
        loudness = maxvol - distanceheardfrom * (getdistance(player,pos)) / 10
        loudness = clamp(loudness,0,maxvol)
        channel.set_volume(loudness)
        spaceawaresndchannels.append((channel, pos, distanceheardfrom, maxvol))

    if sfxtype == "mus":
        channel.set_volume(vol)
        channel.play(sound,loops=loop)
        loudness = maxvol - distanceheardfrom * (getdistance(player,pos)) / 10
        loudness = clamp(loudness,0,maxvol)
        channel.set_volume(loudness)
        spaceawaremuschannels.append((channel, pos, distanceheardfrom,maxvol))


    return channel
def pausesound():
    pygame.mixer.pause()
def unpausesound():
    pygame.mixer.unpause()
def smoothscale(image,size):
    image = pygame.transform.smoothscale(image,size)
    return image
def scaleadd(image, size):
    image = scale(image,(int(image.get_width()+(int(size[0]/2)*2)), int(image.get_height()+(int(size[1]/2)*2))))
    return image
def scale2x(image):
    image = pygame.transform.scale(image,(image.get_width() * 2,image.get_height() * 2))
    return image
def clamp(number, smallest, largest):
    return max(smallest,min(number,largest))
def rect(Rect,color=white,surface=screen,linewidth=0,border_radius=-1):
    pygame.draw.rect(surface,color,Rect,width=linewidth,border_radius=border_radius)
def exit(code=0):
    try:
        quit(code)
    except Exception:
        from sys import exit
        exit(code)
    else:
        pygame.quit()
def loadmap(filepath,filetype="txt"):
    map = []
    file = open(filepath + f".{filetype}","r")
    data = file.read()
    data = data.split('\n')
    file.close()
    for layer in data:
        map.append(list(layer))
    return map
def gradientrect_lefttoright(leftcolour, rightcolour, r, surf=screen, special_flags=0):
    crect = Surface((2,2))
    pygame.draw.line(crect,leftcolour,(0,0),(0,1))
    pygame.draw.line(crect,rightcolour,(1,0),(1,1))
    crect = pygame.transform.smoothscale(crect,(r.width,r.height))
    surf.blit(crect,r, special_flags=special_flags)
def gradientrect_toptobottom(topcolor, bottomcolor, r, surf=screen, special_flags=0):
    crect = Surface((2,2))
    pygame.draw.line(crect,topcolor,(0,0),(1,0))
    pygame.draw.line(crect,bottomcolor,(0,1),(1,1))
    crect = pygame.transform.smoothscale(crect,(r.width,r.height))
    surf.blit(crect,r, special_flags=special_flags)

def getcollisions(tiles,boxcollider):
    return (tile for tile in tiles if tile.colliderect(boxcollider))
def checkcollision(movement,tiles,collider):
    global xscroll,yvelocity,onground,jumping

    collisiontypes = {
        "left": False,
        "right": False,
        "top": False,
        "bottom": False
    }

    if movement != [0,0]:
        if movement[1] != 0:
            collider.y += movement[1]
            collisions = getcollisions(tiles,collider)

            for tile in collisions:
                if movement[1] > 0:
                    collider.bottom = tile.top
                    collisiontypes["bottom"] = True

                if movement[1] < 0:
                    collider.top = tile.bottom
                    collisiontypes["top"] = True
        collider.x += movement[0]
        if movement[0] != 0:
            collisions = getcollisions(tiles,collider)

            for tile in collisions:
                if movement[0] > 0:
                    collider.right = tile.left
                    collisiontypes["right"] = True

                if movement[0] < 0:
                    collider.left = tile.right
                    collisiontypes["left"] = True


    return collisiontypes
def clip(surf,x,y,x_size,y_size):
    handle_surf = surf.copy()
    clipR = pygame.Rect(x,y,x_size,y_size)
    handle_surf.set_clip(clipR)
    image = surf.subsurface(handle_surf.get_clip())
    return image.copy()

def generateambientparticle(pos, color=white, glowcolor=(1,1,1), duration=None, glowiter=None, scrollaffectivity=1, movement=None, fireflybehavior=False, id="all"):
    global ambientparticles

    data = {
        "pos": list((pos[0], pos[1])),
        "color": color,
        "glowcolor": glowcolor,
        "duration":0,
        "maxduration":randint(200, 1000) if duration is None else duration,
        "sine": random.random() * math.pi * 2,
        "state": 0,
        "glowiter": 0,
        "maxglowiter": 7 if glowiter is None else glowiter,
        "scrollaffectivity": scrollaffectivity,
        "movement": [abs(choice(bm)/4), -abs(choice(bm)/4)] if movement is None else movement,
        "fireflybehavior": fireflybehavior,
        "sineadder": random.random() * 0.2 - 0.1,
        "fireflymultiplier": random.random() * 0.25 + 0.1,
        "id": id
    }
    ambientparticles.append(data)
    return data
def generateparticle(pos,img,maxduration=9999999999,size=0,sizechange=0,rotation=0,rotationchange=0,movement=[0,0],movementchange=[0,0],blitincenter=True,alphachange=0,alpha=255,frame=0,frameaddition=1,framestep=5,frames=[],frameindex=0, special_flags=0, id="all"):
    global particles

    data = {
        "id": id,
        "img": img,
        "pos": list(pos),
        "alpha": alpha,
        "index": frameindex,
        "frame": frame,
        "frames": frames,
        "rotation": rotation,
        "size": size,
        "sizechange": sizechange,
        "rotationchange": rotationchange,
        "frameaddition": frameaddition,
        "framekilltimer": 0,
        "framekilldelay": framestep,
        "framestep": framestep,
        "blitcenter": blitincenter,
        "alphachange": alphachange,
        "duration": 0,
        "maxduration": maxduration,
        "movement": list(movement),
        "movementchange": movementchange,
        "specialflags": special_flags
    }
    particles.append(data)

def SoundSpaceHandler(player):
    for snd in spaceawaresndchannels:
        loudness = snd[3] - snd[2] * (getdistance(player,snd[1])) / 10
        loudness = clamp(loudness,0,snd[3])
        snd[0].set_volume(loudness)
        if snd[0].get_busy() == 0:
            spaceawaresndchannels.remove(snd)

def ParticleHandler(scroll=[0,0]):
    global particles, particlekilllist

    r = []

    for particle in particles:
        if particle["id"] in particlekilllist:
            r.append(particle)
        img = particle["img"].copy()
        if particle["alpha"] != 255:
            fillalpha(img,particle["alpha"])

        if particle["size"] != 0:
            img = scale(img,(int(particle["img"].get_width() + particle["size"]),int(particle["img"].get_height() + particle["size"])))

        if particle["rotation"] != 0:
            img = rotate(img,particle["rotation"])

        particle["rotation"] += particle["rotationchange"]
        particle["size"] += particle["sizechange"]

        # blitting
        particle["pos"][0] -= scroll[0]
        particle["pos"][1] -= scroll[1]

        if not particle["blitcenter"]:screen.blit(img,particle["pos"], special_flags=particle["specialflags"])
        if particle["blitcenter"]: blitcenter(img,particle["pos"],special_flags=particle["specialflags"])

        particle["pos"][0] += scroll[0]
        particle["pos"][1] += scroll[1]

        # movement | movement change
        particle["pos"][0] += particle["movement"][0]
        particle["pos"][1] += particle["movement"][1]
        particle["movement"][0] += particle["movementchange"][0]
        particle["movement"][1] += particle["movementchange"][1]

        # frames
        if particle["frames"]:
            maxframe = (particle["framestep"] * len(particle["frames"])) - particle["framestep"]

            if particle["frame"] >= particle["index"] * particle["framestep"] and particle["frame"] != maxframe:
                particle["img"] = particle["frames"][particle["index"]]
                particle["index"] += 1

            if particle["frame"] >= maxframe:
                r.append(particle)
                pass

            if particle["frame"] != maxframe:
                particle["frame"] += particle["frameaddition"]

        # alpha change
        particle["alpha"] += particle["alphachange"]
        if particle["alpha"] <= 1:
            r.append(particle)
            pass

        elif particle["alpha"] > 255:
            r.append(particle)
            pass

        # size
        if img.get_width() <= 0 or img.get_height() <= 0:
            r.append(particle)
            pass

        # duration
        particle["duration"] += 1
        if particle["duration"] == particle["maxduration"]:
            r.append(particle)
            pass

    for p in r:
        try:particles.remove(p)
        except:pass

    particlekilllist.clear()

    return particles
def calculate_movement(angle,speed):
    return [math.cos(angle) * speed * angle,math.sin(angle) * speed]
def getdistance(pos1, pos2):
    distance = math.hypot(pos1[1] - pos2[1], pos1[0] - pos2[0])
    return distance


def dottedline(pos,pos2,width=4,spacing=5,color=white):
    x1, y1 = pos
    x2, y2 = pos2
    diff_x,diff_y = x2 - x1,y2 - y1  # difference vector
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # distance
    norm_diff_x,norm_diff_y = diff_x / dist,diff_y / dist  # normalized difference vector

    total_dots = int(dist // spacing)

    for i in range(total_dots + 1):
        pygame.draw.circle(screen,color,(x1 + norm_diff_x * spacing * i,y1 + norm_diff_y * spacing * i),width / 2)


sparks = []
ambientparticles = []
circleexplosions = []
particles = []
particlekilllist = []
bits = []
bitcache = {}


class BounceVector:
    def __init__(self):
        self.truesize = [0,0]
        self.sizechange = [0,0]

    def update(self):
        self.truesize[0] += self.sizechange[0]
        if self.truesize[0] != 0:
            if self.truesize[0] < 1:
                self.sizechange[0] += 0.7
            if self.truesize[0] > 1:
                self.sizechange[0] -= 0.7
            if self.sizechange[0] < 0:
                self.sizechange[0] += 0.15
            if self.sizechange[0] > 0:
                self.sizechange[0] -= 0.15

        self.truesize[1] += self.sizechange[1]
        if self.truesize[1] != 0:
            if self.truesize[1] < 1:
                self.sizechange[1] += 0.02
            if self.truesize[1] > 1:
                self.sizechange[1] -= 0.02
            if self.sizechange[1] < 0:
                self.sizechange[1] += 0.008
            if self.sizechange[1] > 0:
                self.sizechange[1] -= 0.008

class AmbientEntityManager:
    def __init__(self,imgs,runimgs,runsnd=None,idlesnd=None):
        self.entities = []
        self.imgs = imgs
        self.runimgs = runimgs
        self.runsnd = runsnd
        self.idlesnd = idlesnd

    def createentity(self, pos, squish=True,type=None,framespeed=0.25):
        if not type and len(self.imgs) != 1:type = randint(0, len(self.imgs)-1)
        self.entities.append({"pos":pos, "facing":False,"movement":[0,0], "offset": [0,0], "squish":squish,"size":[0,0],"truesize":[0,0],"type":type,"squisht":0,"state":0,"frametime":0,"framespeed":framespeed,"laststate":0})

    def update(self, scroll, o=0,gm=None,globaltime=0,player=None):
        for entity in self.entities:
            entity["laststate"] = entity["state"]
            if entity["state"] == 0:screen.blit(pygame.transform.flip(self.imgs[entity["type"]],entity["facing"],False) if entity["squisht"] == 0 else scaleadd(pygame.transform.flip(self.imgs[entity["type"]],entity["facing"],False),(4,-4)), ((entity["pos"][0]+o+entity["offset"][0])-scroll[0]+(0 if entity["squisht"] == 0 else -2), entity["pos"][1]+entity["offset"][1]-scroll[1]+(0 if entity["squisht"] == 0 else 2)))
            else:
                screen.blit(pygame.transform.flip(self.runimgs[entity["type"]][int(entity["frametime"])%(len(self.runimgs)-1)],entity["facing"],False),((entity["pos"][0] + o + entity["offset"][0]) - scroll[0] + (0 if entity["squisht"] == 0 else -2),entity["pos"][1] + entity["offset"][1] - scroll[1] + (0 if entity["squisht"] == 0 else 2)))
                entity["frametime"] += entity["framespeed"]

            if gm and entity["state"] == 0:gm.apply_force((entity["pos"][0]+o+(self.imgs[entity["type"]].get_width()/2)+entity["offset"][0],entity["pos"][1]+6+entity["offset"][1]), 5,20)
            if True in (globaltime%randint(120,140)==1, random.random() < 0.004) and entity["offset"][1] == 0 and entity["state"] == 0:
                entity["movement"] = [choice((-2,2)),-2]
                if entity["offset"][0] < 0 and entity["movement"][0] < 0 or entity["offset"][0] > 0 and entity["movement"][0] > 0:entity["movement"][0] *= -1
                if self.idlesnd is not None:
                    if not hasattr(self.idlesnd, '__len__'):playspaceawaresound(self.idlesnd,(entity["pos"][0]+entity["offset"][0],entity["pos"][1]+entity["offset"][1]),player,distanceheardfrom=0.01)
                    else:playspaceawaresound(choice(self.idlesnd),(entity["pos"][0]+entity["offset"][0],entity["pos"][1]+entity["offset"][1]),player,distanceheardfrom=0.01)

            entity["offset"] = addvector(entity["offset"],entity["movement"])
            if entity["state"] == 0:entity["movement"][1] += 0.4
            if entity["movement"][0] != 0:entity["movement"][0] += 0.1 if entity["movement"][0] < 0 else -0.1
            if abs(0.1-entity["movement"][0]) < 0.1:
                entity["movement"][0] = 0

            if entity["movement"][0] != 0:
                entity["facing"] = entity["movement"][0] < 0

            if entity["offset"][1] > 0:
                if entity["movement"][0] != 0:
                    entity["squisht"] = 4
                entity["offset"][1] = 0
                entity["movement"][0] = 0

                if getdistance(player,entity["pos"]) < 70:
                    if entity["state"] == 0:entity["pos"][1] -= 4
                    entity["state"] = 1
                    entity["movement"] = [1 if player[0] < entity["pos"][0]+entity["offset"][0] else -1,-0.5]
                    if self.runsnd is not None:
                        if not hasattr(self.runsnd,'__len__'):playsnd(self.runsnd)
                        else:playsnd(choice(self.runsnd))

            if entity["state"] == 1:
                entity["movement"][1] -= clamp(entity["frametime"]/100,0,0.1)
                entity["movement"][1] = clamp(entity["movement"][1],-7,0)
                entity["movement"][0] += 0.3 if entity["movement"][0] > 0 else -0.3
                entity["movement"][0] = clamp(entity["movement"][0],-5,5)


            if entity["squisht"] != 0:entity["squisht"] -= 1


class GrassManager:
    def __init__(self, imgs, tilesize=32):
        self.cachedgrasstiles = {}
        self.imgs = loadanimation("grass", 5, colorkey=black)
        self.tilesize = tilesize

    def generategrasstile(self, pos, bladecount=None):
        pos = list(pos)
        if bladecount is None:
            bladecount = randint(4,6)

        tile = {"blades": [], "pos": pos}
        lasttype = len(self.imgs)-1

        for i in range(bladecount):
            blade = self.generategrassblade(pos)
            while blade["imgindex"] == lasttype:
                blade = self.generategrassblade(pos)
            lasttype = blade["imgindex"]
            tile["blades"].append(blade)

        self.cachedgrasstiles[f"{tile['pos'][0]},{tile['pos'][1]}"] = tile

    def generategrassblade(self, pos):
        imgindex = randint(0, len(self.imgs)-1)

        blade = {"pos": pos, "imgindex": imgindex,"truerot":0,"rot":0,"stoptimer":35,"lastrot":0}
        return blade

    def updategrass(self,movement,forcepos,scroll,sine):

        for grass in getrenderqueue(self.cachedgrasstiles, scroll):
            offset = 0
            for blade in grass["blades"]:
                if getdistance((forcepos[0] - scroll[0],forcepos[1] - scroll[1]),(blade["pos"][0] + offset - scroll[0],blade["pos"][1] - scroll[1])) < 20 and not -1 < movement[0] < 1:
                    if movement[0] < 0:blade["rot"] = -movement[0] / 1.8 + randint(-1,1)
                    else:blade["rot"] = -movement[0] / 1.8 + randint(-1,1)
                    blade["stoptimer"] = 35

                blade["truerot"] += blade["rot"]
                blade["truerot"] = clamp(blade["truerot"],-60,60)

                if blade["truerot"] != 0:
                    if blade["truerot"] < 1:
                        blade["rot"] += 0.7
                    if blade["truerot"] > 1:
                        blade["rot"] -= 0.7
                    if blade["rot"] < 0:
                        blade["rot"] += 0.15
                    if blade["rot"] > 0:
                        blade["rot"] -= 0.15

                if abs(blade["lastrot"] - blade["truerot"]) < 1:
                    blade["stoptimer"] -= 1
                if blade["stoptimer"] == 0:
                    blade["truerot"] = 0
                    blade["rot"] = 0
                blade["lastrot"] = blade["truerot"]

                img = pygame.transform.rotate(self.imgs[blade["imgindex"]].copy(),clamp(
                    blade["truerot"] + sin(((blade["pos"][0] + offset) / 2) + sine) * 20,-70,70))
                blitcenter(img,(blade["pos"][0] + offset - scroll[0],blade["pos"][1] + 36 - scroll[1]))
                offset += self.tilesize / len(grass["blades"])

def returnNormalisedVector(point1,point2):
    vector = (point2[0] - point1[0],point2[1] - point1[1])
    if vector == (0,0):
        return (0,0)
    magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    nvector = (vector[0] / magnitude,vector[1] / magnitude)
    return nvector
def getrenderqueue(map, pos, tilesize=32):
    #return [map[f"{x + int(round(pos[0] / tilesize - 0.5,0))},{y + int(round(pos[1] / tilesize - 0.5,0))}"] for x in range(math.ceil(1032 / tilesize) + 1) for y in range(math.ceil(532 / tilesize) + 1) if f"{x + int(round(pos[0] / tilesize - 0.5,0))},{y + int(round(pos[1] / tilesize - 0.5,0))}" in map if map[f"{x + int(round(pos[0] / tilesize - 0.5,0))},{y + int(round(pos[1] / tilesize - 0.5,0))}"]["data"]["type"] == "all"]
    """for y in range(math.ceil(532 / tilesize) + 1):
        for x in range(math.ceil(1032 / tilesize) + 1):
            tile_pos = (x + int(round(pos[0] / tilesize - 0.5, 0)), y + int(round(pos[1] / tilesize - 0.5, 0)))

            if f"{tile_pos[0]},{tile_pos[1]}" in map:
                if map[f"{tile_pos[0]},{tile_pos[1]}"]["data"]["type"] == "all":
                    yield map[f"{tile_pos[0]},{tile_pos[1]}"]"""


    renderqueue = []

    for y in range(math.ceil(532 / tilesize) + 1):
        for x in range(math.ceil(1032 / tilesize) + 1):
            tile_pos = (x-1 + int(round(pos[0] / tilesize - 0.5, 0)), y-1 + int(round(pos[1] / tilesize - 0.5, 0)))

            if f"{tile_pos[0]},{tile_pos[1]}" in map:
                if map[f"{tile_pos[0]},{tile_pos[1]}"]["data"]["type"] == "all":
                    renderqueue.append(map[f"{tile_pos[0]},{tile_pos[1]}"])
    return renderqueue

class AnimationManager:
    def __init__(self):
        self.animations = {}
        self.playinganimations = []

        for animation in os.listdir("data/animations"):
            self.animations[animation] = loadanimation(f"data/animations/{animation}", len(os.listdir(f'data/animations/{animation}')))

    def playanimation(self, animationname, framestep=10):
        self.playinganimations.append({"animationname": animationname,"frameindex": len(self.animations[animationname])*10, "framestep": framestep})


class Interactable:
    def __init__(self, pos, img, data):
        self.data = data
        self.img = img
        self.pos = pos

    def update(self,scroll):
        screen.blit(self.img, subvector(self.pos,scroll))


class GlowManager:
    def __init__(self, surf=screen):
        self.glowenabled = False
        self.glowcache = {}
        self.surf = surf
        self.null = Surface((1,1),flags=RLEACCEL)

    def glow(self, iter, pos, glowcolor=(1,1,1), linewidth=0,sep=1):
        if self.glowenabled and glowcolor != (0,0,0):
            iter = int(iter)

            if not f"{iter} {iter}, {glowcolor}" in self.glowcache:
                glowsurf = Surface(((iter*2)*sep,(iter*2)*sep), flags=RLEACCEL)
                for i in range(iter):blitcenter((circle_surf(i * sep,color=glowcolor,linewidth=linewidth)),(glowsurf.get_width() / 2,glowsurf.get_height() / 2),special_flags=BLEND_RGBA_ADD,surf=glowsurf)
                self.glowcache[f"{iter} {iter}, {glowcolor}"] = glowsurf
                blitcenter(glowsurf, pos, special_flags=BLEND_RGB_ADD,surf=self.surf)

            else:blitcenter(self.glowcache[f"{iter} {iter}, {glowcolor}"], pos, special_flags=BLEND_RGB_ADD,surf=self.surf)
        else:return self.null
    def update(self, glowenabled):
        self.glowenabled = glowenabled

def maprect(tile, o, scroll):
    return Rect(((tile["pos"][0] * 32) + o) - scroll[0] + tile["data"]["offset"][0],tile["pos"][1] * 32 - scroll[1] + tile["data"]["offset"][1],tile["data"]["size"][0],tile["data"]["size"][1])
class Mapgenerator:
    def __init__(self, tileimgs):
        self.cachedblocks = []
        self.cachedrenderdata = {}
        self.cachedbgrenderdata = {}
        self.tilekey = {}
        self.tileimgs = tileimgs

    def getbounds(self, map):
        bounds = {
            "top":0,
            "bottom":0,
            "left":0,
            "right":0,
        }

        for tile in map["map"]:
            block = Rect((tile["pos"][0] * 32)+tile["data"]["offset"][0],(tile["pos"][1] * 32)+tile["data"]["offset"][1],self.tileimgs[tile["img"]["s"]][tile["img"]["i"]][0].get_width(), self.tileimgs[tile["img"]["s"]][tile["img"]["i"]][0].get_height())

            if block.left < bounds["left"]:
                bounds["left"] = block.left
            elif block.right > bounds["right"]:
                bounds["right"] = block.right

            if block.top < bounds["top"]:
                bounds["top"] = block.top
            elif block.bottom > bounds["bottom"]:
                bounds["bottom"] = block.bottom

        return bounds

    def cacheblocks(self, map):
        self.tilekey.clear()
        self.cachedblocks.clear()

        for tile in map["map"]:
            if tile["data"]["collision"]:
                block = Rect(tile["pos"][0] * 32,tile["pos"][1] * 32,32,tile["data"]["size"][1])
                self.cachedblocks.append(block)
                self.tilekey[str(tile["pos"])] = block

    def cacherenderdata(self, map):
        self.cachedrenderdata = {}
        self.cachedbgrenderdata = {}

        for tile in filter(lambda t: (t["data"]["type"] != "all"),map["bg"]):
            if not tile["data"]["type"] in self.cachedbgrenderdata:
                self.cachedbgrenderdata[tile["data"]["type"]] = []
            self.cachedbgrenderdata[tile["data"]["type"]].append(tile)

        for tile in filter(lambda t: (t["data"]["type"] != "all"),map["map"]):
            if not tile["data"]["type"] in self.cachedrenderdata:
                self.cachedrenderdata[tile["data"]["type"]] = []
            self.cachedrenderdata[tile["data"]["type"]].append(tile)

    def cacheall(self, map):
        self.cacherenderdata(map)
        self.cacheblocks(map)

    def generateback(self, map, scroll,o=0):
        for tile in filter(lambda t: (t["data"]["type"] == "all"), map["back"]):
            block = Rect(((tile["pos"][0] * 32) + o) - scroll[0] + tile["data"]["offset"][0],tile["pos"][1] * 32 - scroll[1] + tile["data"]["offset"][1],tile["data"]["size"][0],tile["data"]["size"][1])
            block.x += (500-block.x)/32

            if getvisible(block):
                screen.blit(self.tileimgs[tile["img"]["s"]][tile["img"]["i"]][0],block)

    def generatebg(self,map,scroll,o=0):
        for tile in filter(lambda t: (t["data"]["type"] == "all"), map["bg"]):
            block = Rect(tile["pos"][0]+o - scroll[0] + tile["data"]["offset"][0],tile["pos"][1] - scroll[1] + tile["data"]["offset"][1],tile["data"]["size"][0],tile["data"]["size"][1])
            if getvisible(block):screen.blit(self.tileimgs[tile["img"]["s"]][tile["img"]["i"]][0],block)

    def generatemap(self,map,scroll,o=0):
        for tile in filter(lambda t: (t["data"]["type"] == "all"),map["map"]):
            block = maprect(tile,o,scroll)
            if getvisible(block):screen.blit(self.tileimgs[tile["img"]["s"]][tile["img"]["i"]][0],block)

    def generate(self, map, scroll,o=0):
        for tile in filter(lambda t: (t["data"]["type"] == "all" and tile["pos"][0]+o - scroll[0] + tile["data"]["offset"][0]+tile["data"]["size"][0] > 0 and tile["pos"][0]+o - scroll[0] + tile["data"]["offset"][0] < 1000 and tile["pos"][1] - scroll[1] + tile["data"]["offset"][1]+tile["data"]["size"][1] > 0 and tile["pos"][1] - scroll[1] + tile["data"]["offset"][1] < 500), map["back"]):
            screen.blit(self.tileimgs[tile["img"]["s"]][tile["img"]["i"]][0],(tile["pos"][0]+o - scroll[0] + tile["data"]["offset"][0],tile["pos"][1] - scroll[1] + tile["data"]["offset"][1]))

        for tile in filter(lambda t: (t["data"]["type"] == "all"),map["bg"]):
            block = Rect(((tile["pos"][0] * 32)+o) - scroll[0] + tile["data"]["offset"][0],tile["pos"][1] * 32 - scroll[1] + tile["data"]["offset"][1],tile["data"]["size"][0],tile["data"]["size"][1])
            if getvisible(block):screen.blit(self.tileimgs[tile["img"]["s"]][tile["img"]["i"]][0],block)
            block.x += clamp((500-block.x)//10,-32,32)

        for tile in filter(lambda t: (t["data"]["type"] == "all" and tile["pos"][0] + o - scroll[0] + tile["data"]["offset"][0] +tile["data"]["size"][0] > 0 and tile["pos"][0] + o - scroll[0] + tile["data"]["offset"][0] < 1000 and tile["pos"][1] - scroll[1] + tile["data"]["offset"][1] + tile["data"]["size"][1] > 0 and tile["pos"][1] - scroll[1] + tile["data"]["offset"][1] < 500),map["map"]):
            screen.blit(self.tileimgs[tile["img"]["s"]][tile["img"]["i"]][0],((tile["pos"][0] * 32)+o) - scroll[0] + tile["data"]["offset"][0],tile["pos"][1] * 32 - scroll[1] + tile["data"]["offset"][1])


class Spark:
    def __init__(self,loc,angle,speed,color,scale=1,speedchange=0.2,turn=False,glowcolor=(1,1,1),pointtowards=None,reverse=False,antialiasing=False,static=False,type="all", line=False):
        self.loc = loc
        self.antialiasing = antialiasing
        self.pointtowards = pointtowards
        self.glowcolor = glowcolor
        self.reverse = reverse
        self.turn = turn
        self.angle = angle
        self.speed = speed
        self.scale = scale
        self.color = color
        self.speedchange = speedchange
        self.static = static
        self.alive = True
        self.type = type
        self.line = line

    def point_towards(self,angle,rate):
        rotate_direction = ((angle - self.angle + math.pi * 3) % (math.pi * 2)) - math.pi
        try:
            rotate_sign = abs(rotate_direction) / rotate_direction
        except ZeroDivisionError:
            rotate_sing = 1
        if abs(rotate_direction) < rate:
            self.angle = angle
        else:
            self.angle += rate * rotate_sign

    def calculate_movement(self,dt):
        return [math.cos(self.angle) * self.speed * dt,math.sin(self.angle) * self.speed * dt]

    # gravity and friction
    def velocity_adjust(self,friction,force,terminal_velocity,dt):
        movement = self.calculate_movement(dt)
        movement[1] = min(terminal_velocity,movement[1] + force * dt)
        movement[0] *= friction
        self.angle = math.atan2(movement[1],movement[0])
        # if you want to get more realistic, the speed should be adjusted here

    def move(self,dt,update=True):
        if not self.static:
            movement = self.calculate_movement(dt)

            if not self.reverse:
                self.loc[0] += movement[0]
                self.loc[1] += movement[1]

            if self.reverse:
                self.loc[0] -= movement[0]
                self.loc[1] -= movement[1]

        # a bunch of options to mess around with relating to angles...
        if self.pointtowards is not None:
            self.point_towards(self.pointtowards,0.02)

        # self.velocity_adjust(0.975, 0.2, 8, dt)
        if update:
            if self.turn:
                self.angle += 0.025

            self.speed -= self.speedchange

        if self.speed <= 0:
            self.alive = False

    def draw(self,surf,glowmanager):
        if self.alive:
            points = [
                [self.loc[0] + math.cos(self.angle) * self.speed * self.scale,
                 self.loc[1] + math.sin(self.angle) * self.speed * self.scale],
                [self.loc[0] + math.cos(self.angle + math.pi / 2) * self.speed * self.scale * 0.3,
                 self.loc[1] + math.sin(self.angle + math.pi / 2) * self.speed * self.scale * 0.3],
                [self.loc[0] - math.cos(self.angle) * self.speed * self.scale * 3.5,
                 self.loc[1] - math.sin(self.angle) * self.speed * self.scale * 3.5],
                [self.loc[0] + math.cos(self.angle - math.pi / 2) * self.speed * self.scale * 0.3,
                 self.loc[1] - math.sin(self.angle + math.pi / 2) * self.speed * self.scale * 0.3],
            ]

            if self.line:pygame.draw.line(surf,self.color,points[0],points[2], clamp(int(self.speed*self.scale),1,3))
            else:
                if self.antialiasing:
                    gfxdraw.aapolygon(surf,points,self.color)
                    gfxdraw.filled_polygon(surf,points,self.color)
                else:
                    pygame.draw.polygon(surf,self.color,points)

            if self.glowcolor:
                glowmanager.glow((self.speed + 1) * 4,self.loc,glowcolor=self.glowcolor)
class CircleExplosion:
    def __init__(self,pos,radius,linewidth,movement=[0,0],color=white,radiuschange=1,linewidthchange=1, radiuschangechange=0):
        self.pos = pos
        self.radius = radius
        self.radiuschange = radiuschange
        self.radiuschangechange = radiuschangechange
        self.linewidthchange = linewidthchange
        self.linewidth = linewidth
        self.movement = movement
        self.color = list(color)
        self.alive = True

    def update(self):
        self.move()

        #pygame.draw.circle(screen,self.color,self.pos,int(self.radius),int(self.linewidth))
        c = circle_surf(int(self.radius), self.color, int(self.linewidth),transparent=True)
        c.set_alpha(clamp(int(self.linewidth)*15,0,255))
        blitcenter(c, self.pos)

        self.radius -= self.radiuschange
        self.radiuschange -= self.radiuschangechange
        self.linewidth -= self.linewidthchange

        if self.radius <= 1 or self.linewidth <= 1:
            self.alive = False

    def move(self):
        self.pos[0] += self.movement[0]
        self.pos[1] += self.movement[1]
class Throttle:
    def __init__(self,maxspeed,acceleration=0.2,deceleration=0.2):
        self.maxspeed = maxspeed
        self.acceleration = acceleration
        self.deceleration = deceleration
        self.speed = 0

    def update(self,key):
        # state: True=accelerate, False=decelerate

        if key:self.speed += self.acceleration
        if not key:self.speed -= self.deceleration

        self.speed = clamp(self.speed,0,self.maxspeed)

    def reset(self):
        self.speed = 0
class Timer:
    def __init__(self,maxtime,reset=False):
        self.time = maxtime if not reset else 0
        self.maxtime = maxtime

    def tick(self):
        if self.maxtime != self.time:
            self.time += 1

    def reset(self):
        self.time = 0

    def greaterthan(self,number):
        if self.time > number:
            return True
        return False

    def smallerthan(self,number):
        if self.time < number:
            return True
        return False

    def done(self):
        if self.maxtime <= self.time:
            return True
        return False
class ReverseTimer:
    def __init__(self):
        self.time = 0

    def tick(self):
        if self.time != 0:
            self.time -= 1

    def set(self,time):
        self.time = time

    def greaterthan(self,number):
        if self.time > number:
            return True
        return False

    def smallerthan(self,number):
        if self.time < number:
            return True
        return False

    def done(self):
        if self.time <= 0:
            return True
        return False
class FrameClock:
    def __init__(self,maxframe):
        self.frame = 0
        self.maxframe = maxframe

    def tick(self,speed=0.1):
        self.frame += speed
        if int(self.frame) > self.maxframe: self.reset()

    def f(self):
        return int(self.frame)

    def done(self):
        return self.frame >= self.maxframe

    def reset(self):
        self.frame = 0
class Font:
    def __init__(self,path,spacing,color=(255,255,255)):
        self.character_order = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V',
                                'W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
                                's','t','u','v','w','x','y','z','.','-',',',':','+','\'','!','?','0','1','2','3','4',
                                '5','6','7','8','9','(',')','/','_','=','\\','[',']','*','"','<','>',';']
        self.spacing = spacing

        font_img = pygame.image.load(path).convert_alpha()
        font_img = colorswap(font_img,(255,255,255),color)
        font_img.set_colorkey((255,0,0))

        current_char_width = 0
        self.characters = {}
        character_count = 0
        for x in range(font_img.get_width()):
            c = font_img.get_at((x,0))
            if c[0] == 127:
                char_img = clip(font_img,x - current_char_width,0,current_char_width,font_img.get_height())
                self.characters[self.character_order[character_count]] = char_img.copy()
                character_count += 1
                current_char_width = 0
            else:
                current_char_width += 1
        self.space_width = self.characters['A'].get_width()

    def render(self,surf,text,loc):
        x_offset = 0
        for char in text:
            if char != ' ':
                surf.blit(scale2x(self.characters[char]),(loc[0] + x_offset,loc[1]))
                x_offset += self.characters[char].get_width() + self.spacing
            else:
                x_offset += self.space_width + self.spacing
    def renderwavy(self,surf,text,loc, globaltime, frequency=10, multiplier=1):
        wavesine = 0 + globaltime
        x_offset = 0

        for char in text:
            if char != ' ':
                surf.blit(scale2x(self.characters[char]),(loc[0] + x_offset,loc[1]+sin(wavesine/frequency)*multiplier))
                x_offset += self.characters[char].get_width() + self.spacing
            else:
                x_offset += self.space_width + self.spacing
            wavesine += 1

    def getwidth(self,text):
        res = 0
        for char in text:
            res += self.space_width + self.spacing

        return res

def rect_surf(size,radius=0,color=white):
    surf = pygame.Surface((size[0] + radius,size[1] + radius), flags=RLEACCEL|SRCALPHA)
    rect(Rect(0,0,size[0] + radius,size[1] + radius),color=color,surface=surf,border_radius=radius)
    surf.set_colorkey((0,0,0))
    return surf
def ellipse_surf(size,color):
    surf = pygame.Surface((size[0],size[1]),)
    pygame.draw.ellipse(surf,color,Rect(0,0,size[0],size[1]),10)
    surf.set_colorkey((0,0,0))
    return scale2x(surf)
def circle_surf(radius,color=white,linewidth=0,transparent=False):
    if transparent:flags = RLEACCEL|SRCALPHA
    else: flags = RLEACCEL
    surf = pygame.Surface((radius * 2,radius * 2),flags=flags)
    pygame.draw.circle(surf,color,(radius,radius),radius,linewidth)
    return surf
def getnearbytiles(pos,tilekey, tilesize=32):
    rects = []
    tilesize = tilesize
    for offset in [(-3, -3), (-2, -3), (-1, -3), (0, -3), (-3, -2), (-2, -2), (-1, -2), (0, -2), (-3, -1), (-2, -1), (-1, -1), (0, -1), (-3, 0), (-2, 0), (-1, 0), (0, 0), (-2, -2), (-1, -2), (0, -2), (1, -2), (-2, -1), (-1, -1), (0, -1), (1, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (-2, 2), (-1, 2), (0, 2), (1, 2)]:
        searchpos = str([pos[0] // tilesize + offset[0],pos[1] // tilesize + offset[1]])
        if searchpos in tilekey:
            rects.append(tilekey[searchpos])

    return rects
def BitHandler(glowmanager, generator, scroll=(0,0), tilesize=32,bounds={"top":0,"bottom":0,"left":0,"right":0}):
    global bits

    re = []

    for bit in bits:
        r = Rect(int(bit["pos"][0]-1),int(bit["pos"][1]-1), bit["size"],bit["size"])
        if not r.right-scroll[0] < -10 and not r.left-scroll[0] > 1010 and not r.top-scroll[1] > 510 and not r.bottom-scroll[1] < -10:
            if str(bit["color"]) in bitcache:i = bitcache[str(bit["color"])]
            else:
                bitcache[str(bit["color"])] = rect_surf((bit["size"],bit["size"]),color=bit["color"])
                i = bitcache[str(bit["color"])]

            if bit["glow"]: glowmanager.glow(clamp((bit["maxduration"] - bit["duration"]) / 15,10,15),(r.centerx - int(scroll[0]),r.centery - int(scroll[1])),glowcolor=bit["glowcolor"])
            blitcenter(i if bit["alpha"] == 255 else fillalpha(i,bit["alpha"]),(r.centerx - int(scroll[0]),r.centery - int(scroll[1])))

        bit["duration"] += 1
        bit["movement"][0] -= bit["ogmovement"][0]/bit["maxduration"]/1.5
        bit["movement"][1] -= bit["ogmovement"][1]/bit["maxduration"]/1.5
        if isnegative(bit["ogmovement"][0]) != isnegative(bit["movement"][0]):
            bit["movement"][0] = 0

        bit["pos"][0] += bit["movement"][0]
        if bit["physics"]:
            if str([int(bit["pos"][0]//tilesize), int(bit["pos"][1]//tilesize)]) in generator.tilekey or bit["pos"][0] > bounds["right"] or bit["pos"][0] < bounds['left']:
                bit["pos"][0] -= bit["movement"][0]
                bit["movement"][0] *= -0.75
                bit["ogmovement"][0] *= -1
        bit["pos"][1] += clamp(bit["movement"][1],-10,10)
        if bit["physics"]:
            bit["movement"][1] += bit["gravity"]
            if str([int(bit["pos"][0]//tilesize), int(bit["pos"][1]//tilesize)]) in generator.tilekey:
                bit["pos"][1] -= bit["movement"][1]
                bit["movement"][0] *= 0.75
                bit["movement"][1] *= -bit["bounciness"]

        bit["lastpos"] = bit["pos"]
        if bit["duration"] >= bit["maxduration"]:
            re.append(bit)
    for b in re:bits.remove(b)
def AmbientParticleHandler(glowmanager, scroll=(0,0), updateid=["all"]):
    global ambientparticles

    def key(p):return p["scrollaffectivity"]
    for particle in sorted(ambientparticles, key=key):
        r = Rect(((((particle["pos"][0] - scroll[0] * particle["scrollaffectivity"]) / 4) % 300) * 4),(((particle["pos"][1] - scroll[1] * particle["scrollaffectivity"]) / 4) % 200) * 4,2 * particle["scrollaffectivity"],2 * particle["scrollaffectivity"])
        r.size = [clamp(r.size[0],2,5),clamp(r.size[1],2,5)]
        r.x -= 20
        r.y -= 20
        if particle["id"] in updateid:
            alive = True
            if getvisible(r):
                glowmanager.glow(particle["glowiter"] * particle["scrollaffectivity"]+2,r.center,glowcolor=particle["glowcolor"])
                if particle["duration"] == particle["maxduration"] and particle["glowiter"] != 0:screen.blit(fillalpha(rect_surf(r.size,color=particle["color"]), clamp(particle["glowiter"]*5,0,255)), r)
                else:rect(r, color=particle["color"])

            particle["sine"] += particle["sineadder"]
            if random.random() < 0.01: particle["sine"] = random.random() * 0.2 - 0.1
            if particle["glowiter"] != particle["maxglowiter"] and particle["duration"] != particle["maxduration"]: particle["glowiter"] += 1
            if particle["duration"] != particle["maxduration"]: particle["duration"] += 1
            if particle["duration"] == particle["maxduration"] and particle["glowiter"] != 0: particle["glowiter"] -= 1
            if particle["duration"] == particle["maxduration"] and particle["glowiter"] == 0: alive = False
            if not particle["fireflybehavior"]:
                particle["pos"][0] += particle["movement"][0] * particle["scrollaffectivity"]
                particle["pos"][1] += particle["movement"][1] * particle["scrollaffectivity"]
            else:
                if random.random() < 0.01:
                    particle["sineadder"] = random.random() * 0.2 - 0.1
                glowmanager.glow(particle["glowiter"] * particle["scrollaffectivity"],r.center,glowcolor=particle["glowcolor"])

                particle["pos"][0] += cos(particle["sine"]) * particle["fireflymultiplier"] * particle["scrollaffectivity"] * 3
                particle["pos"][1] += sin(particle["sine"]) * particle["fireflymultiplier"] * particle["scrollaffectivity"] * 3

            if not alive: ambientparticles.remove(particle)

def SparkHandler(glowmanager, scroll=(0,0), fixedtype="all"):
    global sparks

    for i,spark in sorted(enumerate(sparks),reverse=True):
        if fixedtype == spark.type:
            spark.move(1)
            o = spark.color

            spark.loc[0] -= scroll[0]
            spark.loc[1] -= scroll[1]
            spark.draw(screen, glowmanager)
            spark.loc[0] += scroll[0]
            spark.loc[1] += scroll[1]

            spark.color = o
            if not spark.alive:
                sparks.pop(i)
def CircleExplosionHandler(scroll=(0,0)):
    global circleexplosions

    r = []

    for circle in circleexplosions:
        circle.pos[0] -= scroll[0]
        circle.pos[1] -= scroll[1]
        circle.update()
        circle.pos[0] += scroll[0]
        circle.pos[1] += scroll[1]

        if not circle.alive:
            r.append(circle)
    for c in r:circleexplosions.remove(c)
def generatebit(pos, color=white, glowcolor=(1,1,1), movement=None, physics=None, duration=None,glow=True, movedt=None, divisor=30,bounciness=0.5,size=2,gravity=0.15):
    global bits, bm

    if movement is None:
        if physics:movement = [choice(bm), choice(bmy)]
        else:movement = [choice(bm)/divisor, choice(bm)/divisor]

    if movedt is None:
        if physics:movedt = 0
        else: movedt = 0

    pos = list(pos)
    pos[0] += movement[0] * movedt
    pos[1] += movement[1] * movedt

    duration = duration if duration != None else randint(100,400)
    physics = choice((False,True,True,True,True)) if physics is None else physics

    data = {
        "pos": pos,
        "lastpos": pos,
        "color": color,
        "glowcolor": glowcolor,
        "movement": list(movement),
        "ogmovement": list(movement),
        "physics": physics,
        "duration":0,
        "maxduration": duration,
        "bounciness":bounciness,
        "glow": glow,
        "size": size,
        "gravity": gravity,
        "alpha": 255
    }

    bits.append(data)
def generatespark(pos,speed=None,angle=None,color=(242,242,242),reverse=False,scale=2,speedchange=0.05, antialiasing=False,turn=False,glowcolor=[1,1,1],move=True,movedt=5,pointtowards=None, line=False,type="all"):
    global sa

    if angle is None:
        angle = choice(sa)

    if speed is None:
        speed = randint(2,4)

    data = Spark(list(pos),math.radians(angle),speed,color,scale=scale,speedchange=speedchange,turn=turn,glowcolor=glowcolor,reverse=reverse,antialiasing=antialiasing,type=type,pointtowards=pointtowards,line=line)
    if move: data.move(movedt,update=False)
    sparks.append(data)
def generatecircleexplosion(pos,radius,linewidth,movement=[0,0],color=white,radiuschange=-2,linewidthchange=0.4,radiuschangechange=0):
    global circleexplosions

    circleexplosions.append(CircleExplosion(list(pos),radius,linewidth,movement=movement,color=color,radiuschange=radiuschange,linewidthchange=linewidthchange, radiuschangechange=radiuschangechange))
def generateshockwave(pos,divisor=90,start=20,max=60):
    global shockwaves

    for i, shockwave in enumerate(shockwaves):
        if shockwave[2] == 10000:
            shockwaves[i][0] = (pos[0] / 500) - 0.5
            shockwaves[i][1] = pos[1] / 500
            shockwaves[i][2] = start
            shockwaves[i][3] = divisor
            shockwaves[i][4] = max
            break

def loadtiles():
    imgtiles = {}
    allsubtiles = os.listdir("data/sprites/tiles")

    for subdir in allsubtiles:
        allsubindir = os.listdir(f"data/sprites/tiles/{subdir}")
        imgtiles[subdir] = []

        for tile in allsubindir:
            filename = f"tiles/{subdir}/{tile}".split(sep=".png")[0]
            if not filename.endswith(".json"):
                configfile = {"colorkey": None,"collision": True,"type": "all","scale": 2,"args": {},"offset": [0,0],"outofbounds": False}

                if os.path.exists(f"data/sprites/tiles/{subdir}/config.json"):
                    cfile = json.load(open(f"data/sprites/tiles/{subdir}/config.json","r"))
                    for data in cfile:configfile[data] = cfile[data]

                if os.path.exists(f"data/sprites/{filename}.json"):
                    cfile = json.load(open(f"data/sprites/{filename}.json","r"))
                    for data in cfile:
                        configfile[data] = cfile[data]

                configfile["size"] = list(loadimage(filename,colorkey=configfile["colorkey"]).get_size())
                img = loadimage(filename,colorkey=configfile["colorkey"])
                imgtiles[subdir].append((img,Throttle(6,acceleration=2,deceleration=2),configfile.copy()))
    return imgtiles

smallfont = Font('data/small font.png',5)
font = Font("data/font.png",10)
imgtiles = loadtiles()
