import os
import pygame
import struct
import moderngl
from data.engine import *
from data.sprites import *
from data.sounds import *
from data.grass import *
from data.psutil import Process
from data.psutil._psutil_windows import REALTIME_PRIORITY_CLASS
try:
    processid = os.getpid()
    p = Process(processid)
    p.nice(REALTIME_PRIORITY_CLASS)
except: pass

VIRTUAL_RES = (1000,500)
ctx = moderngl.create_context()
texture_coordinates = [0, 1, 1, 1,0, 0, 1, 0]
world_coordinates = [-1, -1, 1, -1,-1, 1, 1, 1]
render_indices = [0, 1, 2,1, 2, 3]

prog = ctx.program(
    vertex_shader='''
#version 330
in vec2 vert;
in vec2 in_text;
out vec2 v_text;
void main() {
   gl_Position = vec4(vert, 0.0, 1.0);
   v_text = in_text;
}
''',
    fragment_shader='''
#version 330

in mediump vec2 v_text;                    // screen position
out mediump vec4 f_color;          // fragment output color
uniform float vignettestrength;
uniform float zoom;
uniform vec2 screenshake;
uniform vec4[5] shockwaves;
uniform int shockwavecount;
uniform vec2 scroll;
uniform bool shadow;
uniform float time;

precision mediump float;
uniform sampler2D Texture;
uniform sampler2D NoiseTexture;
const vec2 center = vec2(0.5);

//Blur settings
const float Pi = 6.28318530718; // Pi*2
const float Directions = 16.0;
const float Quality = 2.0;
const float Size = 5.0; // BLUR SIZE (Radius)
const vec2 Radius = Size/vec2(1000,500);
const float ratio = 0.5;
const vec3 bgc = vec3(2, 19, 33) / 255.0;
const vec3 white = vec3(1.0);
const float divisor = (Quality * Directions - 15.0)*4.5;
const vec4 black = vec4(0.0);
const vec2 offset = vec2(0.004);
const vec2 sc = vec2(0.5,0.0);

void main() {
    vec2 v_text2 = center+((v_text - center + screenshake) * 1.0-zoom);

    //shockwave
    vec2 disp = vec2(0.0);
    vec2 scaled_v_text = (v_text - sc) / vec2(ratio,1.0) + sc;

    for (int i = 0; i < shockwavecount; i++) {
        vec4 shockwave = shockwaves[i];
        if (shockwave.z != 10000) {
            float size = shockwave.z/shockwave.a;
            float mask = (1.0 - smoothstep(size-0.1,size,length(scaled_v_text-shockwave.xy))) * smoothstep(size-0.3/(shockwave.z/10.0)-0.15,size-0.3/(shockwave.z/10),length(scaled_v_text-shockwave.xy));
            disp += normalize(scaled_v_text-shockwave.xy) * 0.08/(shockwave.z/7.0) * mask;
        }}
    //shockwave end
    

    //BlUR
    vec4 blurcolor = texture(Texture, v_text2-disp);
    for(float d=0.0; d<Pi; d+=Pi/Directions)
    {
        for (float i=1.0/Quality; i<=1.0; i+=1.0/Quality)
        {blurcolor += texture(Texture, v_text2-disp+vec2(cos(d),sin(d))*Radius*i);}
    }
    blurcolor /= divisor;
    // BLUR END

    float c = (1.0 - length(v_text - center) * vignettestrength);

    vec3 n = texture(NoiseTexture,vec2(v_text2.x-time+scroll.x, v_text2.y+scroll.y)).rgb;
    float noise = clamp(n.r * n.g * n.b * 0.3,0,1);

    f_color = vec4(texture(Texture, v_text2-disp).rgb*c, 1.0)+blurcolor;
    f_color.rgb = mix(f_color.rgb, white, noise);

    if ((shadow) && ((texture(Texture, v_text2 - offset - disp).rgb != bgc) && (texture(Texture, v_text2-disp).rgb == bgc))) {
        f_color = black;
    }
}
''')


screen_texture = ctx.texture(VIRTUAL_RES, 3,pygame.image.tostring(screen, "RGB", True))
noise_texture = ctx.texture(noise.get_size(), 4,pygame.image.tostring(noise, "RGBA", True))
screen_texture.repeat_x = False
screen_texture.repeat_y = False
prog["vignettestrength"] = 0.7
prog["NoiseTexture"] = 1
prog["Texture"] = 0
prog["zoom"] = 0

vbo = ctx.buffer(struct.pack('8f', *world_coordinates))
uvmap = ctx.buffer(struct.pack('8f', *texture_coordinates))
ibo = ctx.buffer(struct.pack('6I', *render_indices))
vao_content = [(vbo, '2f', 'vert'),(uvmap, '2f', 'in_text')]
vao = ctx.vertex_array(prog, vao_content, ibo)
noisedata = noise.get_view('1')
noise_texture.write(noisedata)
noise_texture.use(1)
screen_texture.use()


def render():
    screen_texture.write(screen.get_view('1'))

    vao.render()
    pygame.display.flip()


body = {
    "body": [0,-2],
    "leg1":[0,0],
    "leg2":[0,0],
    "eye1":[0,0],
    "eye2":[0,0]
}
collisiontypes = {
    "left": False,
    "right": False,
    "top": False,
    "bottom": False
}
gamedata = {
    "levelindex": 1,
    "abilities": ["dash","doublejump"],
    "options": {"filter": 1}
}

maps = [json.load(open(f"data/maps/{i}.json")) for i in range(len(os.listdir("data/maps")))]
for map in maps:
    for tile in map["map"]:tile["data"] = imgtiles[tile["img"]["s"]][tile["img"]["i"]][2]
    for tile in map["back"]:tile["data"] = imgtiles[tile["img"]["s"]][tile["img"]["i"]][2]
    for tile in map["bg"]:tile["data"] = imgtiles[tile["img"]["s"]][tile["img"]["i"]][2]

player = Rect(500,220,24,44)
tright = Throttle(4,0.6, 0.6)
tleft = Throttle(4,0.6, 0.6)

generator = Mapgenerator(imgtiles)
grassmanager0 = GrassManager("data/sprites/shortgrass",stiffness=6)
grassmanager = GrassManager("data/sprites/shortgrass",stiffness=6)
grassmanager2 = GrassManager("data/sprites/shortgrass",stiffness=6)
ambientbirdmanager0 = AmbientEntityManager([loadimage(f"bird/{i}") for i in range(3)],[loadanimation(f"birdfly/{i}",1) for i in range(3)],runsnd=sndbirdfly,idlesnd=[sndbirdidle,sndbirdidle2,sndbirdidle3])
ambientbirdmanager = AmbientEntityManager([loadimage(f"bird/{i}") for i in range(3)],[loadanimation(f"birdfly/{i}",1) for i in range(3)],runsnd=sndbirdfly,idlesnd=[sndbirdidle,sndbirdidle2,sndbirdidle3])
ambientbirdmanager1 = AmbientEntityManager([loadimage(f"bird/{i}") for i in range(3)],[loadanimation(f"birdfly/{i}",1) for i in range(3)],runsnd=sndbirdfly,idlesnd=[sndbirdidle,sndbirdidle2,sndbirdidle3])
glowmanager = GlowManager()

if glowmanager.glowenabled:prog["shadow"] = False
else:prog["shadow"] = True

#Les Numeros
zoom = 0
lastc = 0
freezeframe = 0
screenshake = 0
slowmo = 0
globaltime = 0
doublejumptimer = 28
walkframe, walkframe2 = 0, 2.5
transitiondir = 0
cutscenetimer = 0
dashtimer = 0
dashcooltimer = 0
deathtimer = 0
execjump = 0
squish = 0
wallyv = 0
yv = 0
watervol = 0
dashdir = 0
tleftt = 0
trightt = 0
wjm = 0

#Les Listes
lastscroll = [0,0]
playersize = [0,0]
trueplayersize = [0,0]
movement = [0,0]
treeanimations = [((AnimatedFoliage(imgtiles["tree"][i][0], [[38, 92, 66], [62, 137, 72], [99, 199, 77]])), imgtiles["tree"][i][0].get_rect()) for i in range(len(imgtiles["tree"]))]
clouds = [([randint(0,1500),randint(0,470)],randint(8,20)) for i in range(7)]
allbounds = [generator.getbounds(map) for map in maps]
allmaprenderdata = []
allbgrenderdata = []
allgrass = []
allbirds = []
alltrees = []

#level loading
for i, map in enumerate(maps):
    generator.cacherenderdata(map)
    allbgrenderdata.append(generator.cachedbgrenderdata)
    allmaprenderdata.append(generator.cachedrenderdata)

    #grass
    if "grass" in generator.cachedrenderdata:
        for tile in generator.cachedrenderdata["grass"]:
            grassmanager.place_tile((tile["pos"][0],tile["pos"][1]),randint(5,7),tile["data"]["args"])
    if "bird" in generator.cachedbgrenderdata:
        for bird in generator.cachedbgrenderdata["bird"]:
            ambientbirdmanager.createentity(bird["pos"])

    allbirds.append(ambientbirdmanager.entities.copy())
    allgrass.append(grassmanager.grass_tiles.copy())
    grassmanager.grass_tiles.clear()
    ambientbirdmanager.entities.clear()

if gamedata["levelindex"] != 0:grassmanager0.grass_tiles = allgrass[gamedata["levelindex"]-1]
if gamedata["options"]["filter"] == 1: screen_texture.filter = moderngl.NEAREST,moderngl.NEAREST

generator.cacheall(maps[gamedata["levelindex"]])
grassmanager0.grass_tiles = allgrass[gamedata["levelindex"]-1]
grassmanager.grass_tiles = allgrass[gamedata["levelindex"]]
grassmanager2.grass_tiles = allgrass[gamedata["levelindex"]+1]

ambientbirdmanager0.entities.clear()
ambientbirdmanager.entities.clear()
ambientbirdmanager1.entities.clear()
ambientbirdmanager0.entities = allbirds[gamedata["levelindex"]-1]
ambientbirdmanager.entities = allbirds[gamedata["levelindex"]]
ambientbirdmanager1.entities = allbirds[gamedata["levelindex"]+1]
player.center = (allmaprenderdata[gamedata["levelindex"]]["playerspawn"][0]["pos"][0] * 32,allmaprenderdata[gamedata["levelindex"]]["playerspawn"][0]["pos"][1] * 32)
truescroll[0] += (player.centerx - truescroll[0] - 500)
truescroll[1] += (player.centery - truescroll[1] - 280)
if truescroll[0] < allbounds[gamedata["levelindex"]]["left"]:
    truescroll[0] = allbounds[gamedata["levelindex"]]["left"]
if truescroll[0] + 1000 > allbounds[gamedata["levelindex"]]["right"]:
    truescroll[0] = allbounds[gamedata["levelindex"]]["right"] - 1000
if truescroll[1] > allbounds[gamedata["levelindex"]]["bottom"] - 500:
    truescroll[1] = allbounds[gamedata["levelindex"]]["bottom"] - 500

#Les Booles
dying = False
facing = False
dash = False
inwater = False
lastinwater = False
transitioning = False
movementlock = False
doublejumpable = False
inwind = False

for i in range(16):generateambientparticle([randint(0,1000),randint(0,500)],glowiter=10,glowcolor=(1,1,1),duration=float("inf"),scrollaffectivity=choice([1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]))
for i in range(8):generateambientparticle([randint(0,1000),randint(0,500)],glowiter=10,glowcolor=(1,1,1),duration=float("inf"),scrollaffectivity=choice([0.5,0.6,0.7,0.8,0.9]),id="bg")


def removescroll(vector):
    global scroll

    vector = [vector[0] + scroll[0], vector[1] + scroll[1]]

    return vector
def addscroll(vector):
    global scroll

    vector = [vector[0] - scroll[0], vector[1] - scroll[1]]

    return vector

rot_function = lambda x,y: int(math.sin(globaltime / 20 + x / 100) * 6)
waterchannel = playsnd(sndwater,loop=9999999)
pygame.event.clear(gameupdate)
playsnd(sndambience)




while 1:
    keys = Keys()
    for event in Events():
        if event.type == QUIT:
            exit()
        if event.type == KEYDOWN:
            if event.key == K_z and not transitioning and not movementlock:
                if "doublejump" in gamedata["abilities"] and not collisiontypes["bottom"] and not ("walljump" in gamedata["abilities"] and True in (collisiontypes["left"], collisiontypes["right"])) and doublejumpable and lastc == 0:
                    doublejumptimer = 0
                    execjump = 0

                if collisiontypes["bottom"] or lastc != 0:
                    trueplayersize = addvector((-6,6),trueplayersize)
                    playersize = [-16,16]
                    execjump = 0
                    yv = -7.5

                if "walljump" in gamedata["abilities"] and collisiontypes["left"] and not collisiontypes["bottom"]:
                    playersize = [-16,16]
                    execjump = 0
                    yv = -7
                    tleft.speed = 0
                    tright.speed = 5
                    wjm = 5

                elif "walljump" in gamedata["abilities"] and collisiontypes["right"] and not collisiontypes["bottom"]:
                    playersize = [-16,16]
                    execjump = 0
                    yv = -7
                    tleft.speed = 5
                    tright.speed = 0
                    wjm = -5

                else:execjump = 3

            if event.key == K_x and "dash" in gamedata["abilities"] and dash and not transitioning and not movementlock:
                if tleftt < trightt and tleftt != trightt:
                    dashdir = 1
                if tleftt > trightt and tleftt != trightt:
                    dashdir = -1
                if tleftt == trightt:
                    dashdir = 1 if facing else -1

                player.topleft = addscroll(player.topleft)
                generateshockwave(removescroll(player.center),divisor=120,max=100,start=28)
                player.topleft = removescroll(player.topleft)
                playsnd(snddash)
                screenshake = 5
                dashtimer = 6
                dash = False

        if event.type == gameupdate:
            screen.fill((2, 19, 33))

            if keys[K_LEFT]:tleftt += 1
            else:tleftt = 0
            if keys[K_RIGHT]:trightt += 1
            else:trightt = 0

            if not transitioning:
                globaltime += 1

                if not movementlock:
                    tleft.update(keys[K_LEFT])
                    tright.update(keys[K_RIGHT])
                else:
                    tleft.update(False)
                    tright.update(False)
                if not movementlock:
                    possible = False

                    if dashtimer != 0:
                        generatespark((player.centerx+randint(-2,2),player.centery+randint(-10,10)),angle=180 if facing else 0,line=True,speedchange=0.2,scale=1,glowcolor=(0,0,0))
                        if dashtimer in (6,3):generateparticle(player.center,imgplayer,alpha=100,alphachange=-7)

                    if execjump != 0 and collisiontypes["bottom"] or execjump != 0 and lastc != 0 or dashtimer != 0 and possible and execjump != 0:
                        playersize = [-16,16]
                        execjump = 0
                        yv = -7.5
                    if execjump != 0 and "walljump" in gamedata["abilities"] and collisiontypes["left"]:
                        playersize = [-16,16]
                        execjump = 0
                        yv = -7.5
                        tleft.speed = 4
                        tright.speed = 0
                    if execjump  != 0 and "walljump" in gamedata["abilities"] and collisiontypes["right"]:
                        playersize = [-16,16]
                        execjump = 0
                        yv = -7.5
                        tleft.speed = 0
                        tright.speed = 4

                if yv < 0 and not keys[K_z]:yv += (0-yv)/2
                if execjump != 0:execjump -= 1

            if dashtimer != 0 and not transitioning:
                dashtimer -= 1
                yv = 0
                wjm = 0
                if dashtimer == 0:
                    dashcooltimer = (14*dashdir) if not inwater else (9*dashdir)
                    if dashdir == -1:tleft.speed = 5
                    if dashdir == 1:tright.speed = 5

            if dashcooltimer != 0 and not transitioning:
                if abs(dashcooltimer) > 3:yv = 0
                if dashcooltimer > 0:dashcooltimer -= 1
                else:dashcooltimer += 1
            if wjm != 0 and not transitioning:
                if wjm > 0:wjm -= 1
                else:wjm += 1
            if abs(dashcooltimer) == 12 and not inwater: generateparticle(player.center,imgplayer,alpha=100,alphachange=-7)
            if dashtimer == 0:
                dashdir = 0

            if dashtimer == 0:movement = [(tright.speed-tleft.speed)+dashcooltimer+wjm,yv]
            else:movement = [(14*dashdir) if not inwater else (9*dashdir),0]
            if not inwater and movement[0] != 0 and random.random() < 0.28 and collisiontypes["bottom"]:#smoke
                generateparticle(player.midbottom,imgsmoke,alphachange=-2,sizechange=1,alpha=40,size=-11,movementchange=[0,-0.05])
            if not dash and dashtimer == 0 and dashcooltimer == 0:generateparticle((player.centerx+randint(-10,10),player.centery+randint(-10,10)),imgsmoke,alphachange=-2,sizechange=1,alpha=40,size=-11,movementchange=[0,-0.05])
            if movement[0] != 0:facing = ispositive(movement[0])
            prectypes = collisiontypes.copy()

            if not transitioning:collisiontypes = checkcollision(movement,getnearbytiles(player.center,generator.tilekey),player)
            if collisiontypes["top"]:
                yv = 0
            if player.top > allbounds[gamedata["levelindex"]]["bottom"] and not dying:
                for i in range(30):
                    generatespark(player.center,move=True,movedt=15)
                    generatebit(player.center,physics=True,movement=[choice(bm)*3,choice(bmy)*3])
                    generateparticle(player.center,imgsmoke,movement=[cos(i)*5,sin(i)*5],movementchange=[-cos(i)*0.1,-sin(i)*0.1],sizechange=-0.5,size=10)

                generateshockwave(player.center,max=120)
                movementlock = True
                deathtimer = 0
                shockwave = 10
                dying = True
            if "walljump" in gamedata["abilities"]:
                if True in (collisiontypes["left"],collisiontypes["right"]) and yv > 0:
                    wallyv += 0.1
                    yv = wallyv
                if not True in (collisiontypes["left"],collisiontypes["right"]):wallyv = 0.5


            if not prectypes["bottom"] and collisiontypes["bottom"]:
                for i in range(int(yv)):generatebit(player.midbottom,physics=True,bounciness=0.2,gravity=0.2)
                trueplayersize = addvector((8,-8),trueplayersize)
                playersize = [yv*2.5,-yv*2.5]
                playsnd(sndsquish,vol=yv/10)
                squish = 6


            if squish and not transitioning and not movementlock:body["body"][1] += squish*1.5
            squish -= 1 if squish != 0 else 0
            if player.right-1 < allbounds[gamedata["levelindex"]]["left"]:
                player.right = allbounds[gamedata["levelindex"]]["left"]+1
                transitioning = True
                transitiondir = -1
            if player.left+1 > allbounds[gamedata["levelindex"]]["right"]:
                player.left = allbounds[gamedata["levelindex"]]["right"]-1
                transitioning = True
                transitiondir = 1
            if gamedata["levelindex"] == 0:
                if player.left < allbounds[gamedata["levelindex"]]["left"]:
                    player.left = allbounds[gamedata["levelindex"]]["left"]


            if dying:
                deathtimer += 1
                if deathtimer == 100:
                    player.center = (allmaprenderdata[gamedata["levelindex"]]["playerspawn"][0]["pos"][0] * 32,allmaprenderdata[gamedata["levelindex"]]["playerspawn"][0]["pos"][1] * 32)
                    movementlock = False
                    dash = True
                    doublejumpable = True
                    dying = False

            if collisiontypes["left"] and tleft.speed > 1:tleft.speed = 1
            if collisiontypes["right"] and tright.speed > 1:tright.speed = 1
            if not collisiontypes["bottom"] and not transitioning:
                tleft.acceleration = 0.4
                tleft.deceleration = 0.5
                tright.acceleration = 0.4
                tright.deceleration = 0.5
                tleft.maxspeed = 5
                tright.maxspeed = 5
                lastc -= 1 if lastc != 0 else 0
                if not inwater:
                    if yv < 0 or inwind:yv += 0.35
                    else:yv += 0.4
                else:yv += 0.15


            elif collisiontypes["bottom"]:
                tleft.acceleration = 0.6
                tleft.deceleration = 0.5
                tright.acceleration = 0.6
                tright.deceleration = 0.5
                tleft.maxspeed = 4
                tright.maxspeed = 4
                wallyv = 0
                lastc = 3
                yv = 1
                dash = True
                doublejumpable = True

            if yv > 10:yv = 10
            scrolltarget = player.center
            if movementlock:
                cutscenetimer += 1
            if not transitioning:
                body["eye1"][0] += (movement[0]-body["eye1"][0])/5
                body["eye2"][0] += (movement[0]-body["eye2"][0])/5
                body["eye1"][0] = clamp(body["eye1"][0],-4,4)
                body["eye2"][0] = clamp(body["eye2"][0],-4,4)

                body["eye1"][1] += (movement[1]-body["eye1"][1])/5
                body["eye2"][1] += (movement[1]-body["eye2"][1])/5
                body["eye1"][1] = clamp(body["eye1"][1],-4,4)
                body["eye2"][1] = clamp(body["eye2"][1],-4,4)

                if movement[0] == 0 and collisiontypes["bottom"]:
                    if not movementlock:body["body"][1] += ((sin(globaltime/20)*2)-(body["body"][1]-2))/2
                    body["leg1"][1] += 0-(body["leg1"][1])//3
                    body["leg2"][1] += 0-(body["leg2"][1])//3
                if movement[0] != 0 and collisiontypes["bottom"]:
                    walkframe += 0.25
                    walkframe2 += 0.25
                    if not movementlock:body["body"][1] += ((sin(globaltime/2)*2)-(body["body"][1]-2))/2
                    body["leg1"][1] += -(0 + sin(walkframe) * 2.2)-(body["leg1"][1]+4)//2
                    body["leg2"][1] += -(0 + sin(walkframe2) * 2.2)-(body["leg2"][1]+4)//2
                if not collisiontypes["bottom"]:
                    if not movementlock:body["body"][1] += 0-(body["body"][1]-1)/2
                    body["leg1"][1] += 0-(body["leg1"][1])//3
                    body["leg2"][1] += 0-(body["leg2"][1])//3

            for cloud in clouds:
                if 8 <= cloud[1] < 12:
                    i = imgclouds[0],0.3
                if 12 <= cloud[1] < 16:
                    i = imgclouds[1],0.2
                if 16 <= cloud[1] <= 20:
                    i = imgclouds[2],0.1

                l = [((cloud[0][0]-(scroll[0]/20)) % 1200) - 150,cloud[0][1]]
                l[1] -= scroll[1]/20
                screen.blit(i[0],l)
                cloud[0][0] -= i[1]

            generator.generateback(maps[gamedata["levelindex"]],scroll)
            generator.generatebg(maps[gamedata["levelindex"]],scroll)
            generator.generateback(maps[gamedata["levelindex"]+1],scroll,allbounds[gamedata["levelindex"]]["right"])
            generator.generatebg(maps[gamedata["levelindex"]+1],scroll,allbounds[gamedata["levelindex"]]["right"])
            generator.generateback(maps[gamedata["levelindex"]-1],scroll,-allbounds[gamedata["levelindex"]-1]["right"])
            generator.generatebg(maps[gamedata["levelindex"]-1],scroll,-allbounds[gamedata["levelindex"]-1]["right"])

            AmbientParticleHandler(glowmanager,scroll,updateid="bg")

            for levels in range(-1,2):
                if "tree" in allbgrenderdata[gamedata["levelindex"]+levels]:
                    for tree in allbgrenderdata[gamedata["levelindex"]+levels]["tree"]:
                        treer = treeanimations[tree["img"]["i"]][1]
                        treer.topleft = (tree["pos"][0]-scroll[0],tree["pos"][1]-scroll[1])
                        if getvisible(treer):treeanimations[tree["img"]["i"]][0].render(screen,treer,globaltime/80)

                if "lamp" in allbgrenderdata[gamedata["levelindex"]+levels]:
                    for lamp in allbgrenderdata[gamedata["levelindex"]+levels]["lamp"]:
                        lampr = imgtiles["ruins"][lamp["img"]["i"]][0].get_rect(topleft=(lamp["pos"][0]-scroll[0], lamp["pos"][1]-scroll[1]))
                        if levels == -1:lampr.x -= allbounds[gamedata["levelindex"] - 1]["right"]
                        if levels == 1:lampr.x += allbounds[gamedata["levelindex"]]["right"]

                        if getvisible(lampr, extraspace=45):
                            glowmanager.glow(25,(lampr.x + 13,lampr.y + 15),glowcolor=(4,3,0),sep=3)
                            screen.blit(imgtiles["ruins"][lamp["img"]["i"]][0], lampr)


                inwind = False
                if "windstream" in allmaprenderdata[gamedata["levelindex"]+levels]:
                    for windstream in allmaprenderdata[gamedata["levelindex"]+levels]["windstream"]:
                        windstreamr = Rect((windstream["pos"][0]*32)-scroll[0],(windstream["pos"][1]*32)-scroll[1],32,32)

                        player.center = addscroll(player.center)
                        if player.colliderect(windstreamr):
                            inwind = True
                            yv -= 0.5
                            yv = clamp(yv, -9,10)
                            playersize = [-12,12]
                        if random.random() < 0.02:
                            generatespark(addvector(removescroll(windstreamr.midbottom),(randint(-16,16),randint(-5,24))),line=True,angle=-90,glowcolor=(0,0,0),color=choice(((200,200,210),(150,150,170),(100,100,150))),speed=5)
                        player.center = removescroll(player.center)

                if "crystal" in allmaprenderdata[gamedata["levelindex"]+levels]:
                    for crystal in allmaprenderdata[gamedata["levelindex"]+levels]["crystal"]:
                        crystalr = imgtiles["collectibles"][crystal["img"]["i"]][0].get_rect(topleft=((crystal["pos"][0]*32)-scroll[0], (crystal["pos"][1]*32)-scroll[1]+(sin(globaltime/20)*2)))
                        if levels == -1:crystalr.x -= allbounds[gamedata["levelindex"] - 1]["right"]
                        if levels == 1:crystalr.x += allbounds[gamedata["levelindex"]]["right"]

                        if globaltime%100==1 and crystal["data"]["args"]["state"] == 0:
                            generateshockwave(removescroll(crystalr.center),divisor=80,max=100,start=20.5)

                        if getvisible(crystalr):
                            if crystal["data"]["args"]["state"] != 2:
                                glowmanager.glow(24,crystalr.center,glowcolor=(1,1,3),sep=2)
                                screen.blit(imgtiles["collectibles"][crystal["img"]["i"]][0],crystalr)
                            crystalr.topleft = removescroll(crystalr.topleft)

                            if crystal["data"]["args"]["state"] == 0 and random.random() < 0.2:
                                generatespark(crystalr.center,color=(70,150,255),glowcolor=(1,1,3),speed=randint(1,2),speedchange=0.08,movedt=20)

                            if player.colliderect(crystalr) and crystal["data"]["args"]["state"] == 0:
                                for i in range(30):
                                    generatebit(crystalr.center,(164, 241, 255), glowcolor=(1,1,3),physics=True,gravity=0.2,movement=[choice(bm)*2,choice(bmy)*2],bounciness=0.4)
                                    generatespark(crystalr.center,color=(70, 150, 255), glowcolor=(1,1,2),speed=randint(4,6),speedchange=0.08,scale=1)
                                crystal["data"]["args"]["state"] = 1
                                movementlock = True
                                slowmo = 9

                            if crystal["data"]["args"]["state"] == 1:
                                crystal["data"]["args"]["ct"] += crystal["data"]["args"]["cv"]
                                crystal["data"]["args"]["cv"] -= 0.01
                                if cutscenetimer == 9:
                                    generateshockwave(crystalr.center,divisor=80,max=100,start=20.5)
                                    crystal["data"]["args"]["state"] = 2
                                    screenshake = 10

                            if cutscenetimer != 100 and "walljump" not in gamedata["abilities"] and crystal["data"]["args"]["state"] == 2:
                                crystal["data"]["args"]["ct"] += crystal["data"]["args"]["cv"]
                                crystal["data"]["args"]["cv"] -= 0.005

                            if cutscenetimer > 100 and "walljump" not in gamedata["abilities"]:
                                gamedata["abilities"].append("walljump")
                                movementlock = False
                                cutscenetimer = 0

            for bird in ambientbirdmanager.entities:
                if bird["state"] == 1 and bird["offset"][1] > -1500 and getvisiblefrompos((bird["pos"][0]+bird["offset"][0]-scroll[0],bird["pos"][1]+bird["offset"][1]-scroll[1])):
                    generateparticle(((bird["pos"][0] + bird["offset"][0]) + (0 if bird["squisht"] == 0 else -2),bird["pos"][1] + bird["offset"][1] + (0 if bird["squisht"] == 0 else 2)),pygame.transform.flip(ambientbirdmanager.runimgs[bird["type"]][int(bird["frametime"]) % (len(ambientbirdmanager.runimgs) - 1)],bird["facing"],False),alphachange=-20,alpha=80,blitincenter=False)
                    #generatebit(((bird["pos"][0] + bird["offset"][0]+12+randint(-4,4)) + (0 if bird["squisht"] == 0 else -2),bird["pos"][1]+randint(-4,4) + bird["offset"][1] + 12 + (0 if bird["squisht"] == 0 else 2)),color=ambientbirdmanager.imgs[bird["type"]].get_at((10,10)),physics=False,duration=randint(10,20),glowcolor=[e / 200 for e in ambientbirdmanager.imgs[bird["type"]].get_at((10,10))])

            ambientbirdmanager0.update(scroll,gm=grassmanager,globaltime=globaltime,player=player.center,o=-allbounds[gamedata["levelindex"]-1]["right"])
            ambientbirdmanager.update(scroll,gm=grassmanager,globaltime=globaltime,player=player.center)
            ambientbirdmanager1.update(scroll,gm=grassmanager,globaltime=globaltime,player=player.center,o=allbounds[gamedata["levelindex"]]["right"])

            if "camlock" in allmaprenderdata[gamedata["levelindex"]]:
                for camlock in allmaprenderdata[gamedata["levelindex"]]["camlock"]:
                    camlockr = Rect((camlock["pos"][0]*32)-scroll[0], (camlock["pos"][1]*32)-scroll[1], 32, 32)
                    if getdistance(addscroll(player.center),camlockr.center) < 250:
                        scrolltarget = removescroll(camlockr.center)

            ParticleHandler(scroll)
            player.topleft = addscroll(player.topleft)
            glowmanager.glow(30,player.center,sep=3)


            if doublejumptimer < 8:yv /= 1.15
            if doublejumptimer == 8:
                trueplayersize = addvector((-6,6),trueplayersize)
                doublejumpable = False
                playersize = [-20,20]
                playsnd(snddoublejump)
                yv = -9.5
            if doublejumptimer <= 28:
                generateparticle((player.centerx+scroll[0]+choice((-30,30))+randint(-10,10),player.centery+scroll[1]+randint(-10,16)),colorize(imgfeathers[0],white),rotationchange=2,rotation=randint(0,360),alphachange=-5,movement=[0,-1],movementchange=[0,0.05],sizechange=-0.5)
                generateparticle(removescroll(player.center),imgwings[int(doublejumptimer / 4)],alphachange=-50,alpha=200)
                blitcenter(imgwings[int(doublejumptimer / 4)],player.center)
                doublejumptimer += 1


            t = round(trueplayersize[0]), round(trueplayersize[1])
            blitcenter(scaleadd(imgbody["body"],t),(player.x + body["body"][0] + 12,player.y + body["body"][1] + 17))
            blitcenter(scaleadd(imgbody["leg"],(t[0] // 3,t[1] // 3)),(player.x + body["leg1"][0] + 4 - round(t[0] / 3),player.y + player.height - 10 + body["leg1"][1] - 5 + 8))
            blitcenter(scaleadd(imgbody["leg"],(t[0] // 3,t[1] // 3)),(player.x + body["leg2"][0] + 4 + round(t[0] / 3) + (player.width // 3) * 2,player.y + player.height - 10 + body["leg2"][1] - 5 + 8))
            rect(Rect(player.x+8+body["eye1"][0],player.y+10+body["body"][1]+body["eye1"][1],2,6),black)
            rect(Rect(player.x+14+body["eye2"][0],player.y+10+body["body"][1]+body["eye2"][1],2,6),black)
            player.topleft = removescroll(player.topleft)


            BitHandler(glowmanager,generator,scroll,bounds=allbounds[gamedata["levelindex"]])

            #water
            for levels in range(-1,2):
                if "water" in allmaprenderdata[gamedata["levelindex"]+levels]:
                    if levels == 0:
                        lastinwater = inwater
                        inwater = False
                    for water in allmaprenderdata[gamedata["levelindex"]+levels]["water"]:
                        waterr = Rect((water["pos"][0]*32)-scroll[0],(water["pos"][1]*32)-scroll[1],32,32)
                        if levels == -1:waterr.x -= allbounds[gamedata["levelindex"] - 1]["right"]
                        if levels == 1:waterr.x += allbounds[gamedata["levelindex"]]["right"]

                        if getvisible(waterr):
                            player.topleft = addscroll(player.topleft)

                            if player.colliderect(waterr):
                                tleft.acceleration = 0.3
                                tright.acceleration = 0.3
                                tleft.deceleration = 0.4
                                tright.deceleration = 0.4
                                tleft.maxspeed = 3
                                tright.maxspeed = 3
                                inwater = True

                                #splash
                                if inwater and not lastinwater:
                                    playsnd(sndwaterin)
                                    yv /= 1.8
                                    for i in range(int(yv)):generatebit(removescroll((player.centerx,waterr.y+2)),physics=True,color=white,glow=False,movement=[choice(bm), clamp(-abs(choice(bmy)),-4,-1)])
                                yv = clamp(yv,-5,6)

                            player.topleft = removescroll(player.topleft)
                            screen.blit(imgwater[int((globaltime/3)+(water["pos"][0]*2))%22],waterr.topleft)

            if lastinwater and not inwater:
                playsnd(sndwaterout)

            grassmanager.apply_force(player.center,10,30)
            grassmanager0.update_render(screen,1,(scroll[0]+allbounds[gamedata["levelindex"]-1]["right"],scroll[1]-2),rot_function)
            grassmanager2.update_render(screen,1,(scroll[0]-allbounds[gamedata["levelindex"]]["right"],scroll[1]-2),rot_function)
            grassmanager.update_render(screen,1,(scroll[0],scroll[1]-2),rot_function)
            generator.generatemap(maps[gamedata["levelindex"]+1],(scroll[0],scroll[1]),allbounds[gamedata["levelindex"]]["right"])
            generator.generatemap(maps[gamedata["levelindex"]-1],(scroll[0],scroll[1]),-allbounds[gamedata["levelindex"]-1]["right"])
            generator.generatemap(maps[gamedata["levelindex"]],scroll,0)

            CircleExplosionHandler(scroll)
            SparkHandler(glowmanager,scroll)
            AmbientParticleHandler(glowmanager,scroll)

            playersize[0] -= (player.width + playersize[0] - player.width) / 5
            playersize[1] -= (player.height + playersize[1] - player.height) / 5
            trueplayersize[0] += (playersize[0] - trueplayersize[0]) / 2
            trueplayersize[1] += (playersize[1] - trueplayersize[1]) / 2
            prog["scroll"] = (scroll[0]/800,scroll[1]/300)


            if not transitioning:
                truescroll[0] += (scrolltarget[0] - truescroll[0] - 500) / 20
                truescroll[1] += (scrolltarget[1] - truescroll[1] - 280) / 30
            else:
                if transitiondir == 1:
                    if (allbounds[gamedata["levelindex"]]["right"]+500) - truescroll[0] - 500 > 10:truescroll[0] += ((allbounds[gamedata["levelindex"]]["right"]+500) - truescroll[0] - 500) / 10
                    else:truescroll[0] += ((allbounds[gamedata["levelindex"]]["right"]+500) - truescroll[0] - 500) / 3
                else:
                    if (allbounds[gamedata["levelindex"]]["left"] - 500) - truescroll[0] - 500 < 10:truescroll[0] += ((allbounds[gamedata["levelindex"]]["left"] - 500) - truescroll[0] - 500) / 9
                    else:truescroll[0] += ((allbounds[gamedata["levelindex"]]["left"] - 500) - truescroll[0] - 500) / 2


            if transitioning:
                if transitiondir == 1 and int((allbounds[gamedata["levelindex"]]["right"]+500) - truescroll[0] - 500) == 0:
                    for particle in ambientparticles:particle["pos"][0] -= (scroll[0]*particle["scrollaffectivity"])
                    for cloud in clouds:cloud[0][0] -= (allbounds[gamedata["levelindex"]]["right"]/20)
                    for shockwave in shockwaves:shockwave = [0,0,10000,100,1000]
                    bits.clear()
                    sparks.clear()
                    transitiondir = 0
                    transitioning = False
                    gamedata["levelindex"] += 1
                    generator.cacheall(maps[gamedata["levelindex"]])
                    player.x = allbounds[gamedata["levelindex"]]["left"]
                    truescroll[0] = allbounds[gamedata["levelindex"]]["left"]
                    grassmanager0.grass_tiles = allgrass[gamedata["levelindex"] - 1]
                    grassmanager.grass_tiles = allgrass[gamedata["levelindex"]]
                    grassmanager2.grass_tiles = allgrass[gamedata["levelindex"] + 1]
                    ambientbirdmanager0.entities = allbirds[gamedata["levelindex"] - 1].copy()
                    ambientbirdmanager.entities = allbirds[gamedata["levelindex"]].copy()
                    ambientbirdmanager1.entities = allbirds[gamedata["levelindex"] + 1].copy()


                if transitiondir == -1 and int((allbounds[gamedata["levelindex"]]["left"]-500) - truescroll[0] - 500) == 0:
                    for particle in ambientparticles:particle["pos"][0] += (allbounds[gamedata["levelindex"]-1]["right"]*particle["scrollaffectivity"])
                    for cloud in clouds:cloud[0][0] += (allbounds[gamedata["levelindex"]-1]["right"]/20)
                    for shockwave in shockwaves:shockwave = [0,0,10000,100,1000]
                    bits.clear()
                    sparks.clear()
                    transitiondir = 0
                    transitioning = False
                    gamedata["levelindex"] -= 1
                    generator.cacheall(maps[gamedata["levelindex"]])
                    player.right = allbounds[gamedata["levelindex"]]["right"]
                    truescroll[0] = allbounds[gamedata["levelindex"]]["right"]
                    grassmanager0.grass_tiles = allgrass[gamedata["levelindex"] - 1]
                    grassmanager.grass_tiles = allgrass[gamedata["levelindex"]]
                    grassmanager2.grass_tiles = allgrass[gamedata["levelindex"] + 1]
                    ambientbirdmanager0.entities = allbirds[gamedata["levelindex"] - 1].copy()
                    ambientbirdmanager.entities = allbirds[gamedata["levelindex"]].copy()
                    ambientbirdmanager1.entities = allbirds[gamedata["levelindex"] + 1].copy()


            if truescroll[0] < allbounds[gamedata["levelindex"]]["left"] and not (transitioning and transitiondir == -1):
                truescroll[0] = allbounds[gamedata["levelindex"]]["left"]
            if truescroll[0]+1000 > allbounds[gamedata["levelindex"]]["right"] and not (transitioning and transitiondir == 1):
                truescroll[0] = allbounds[gamedata["levelindex"]]["right"]-1000
            if truescroll[1] > allbounds[gamedata["levelindex"]]["bottom"]-500:
                truescroll[1] = allbounds[gamedata["levelindex"]]["bottom"]-500
            scroll = [round(truescroll[0]), round(truescroll[1])]


            if screenshake != 0:
                if screenshakeinterval == 1:
                    screenshake -= 1
                    screenshakeinterval = 0
                else:screenshakeinterval += 1

            SoundSpaceHandler(player.center)
            prog["zoom"] = clamp(zoom,0,100)
            if screenshake != 0:
                prog["screenshake"] = ((choice([-1,1]) * abs(screenshake/1.5))/1000, (choice([-1,1]) * abs(screenshake/1.5))/500)
            else:prog["screenshake"] = (0,0)

            if freezeframe != 0:
                slowmotion(freezeframe)
                freezeframe = 0
            if slowmo != 0:
                slowmotion(2)
                slowmo -= 1

            if inwater:watervol += (0.4-watervol) / 5
            else:watervol += (0-watervol) / 5

            waterchannel.set_volume(watervol)
            for i, shockwave in enumerate(shockwaves):
                shockwaves[i][2] += 1
                if shockwave[2] >= shockwave[4]:shockwaves[i] = [0,0,10000,100,100]
            prog["shockwaves"] = [(shockwave[0]-(scroll[0]/500),shockwave[1]-(scroll[1]/500),shockwave[2],shockwave[3]) for shockwave in shockwaves]
            prog["shockwavecount"] = len(shockwaves)
            prog["time"] = globaltime/1800

    render()