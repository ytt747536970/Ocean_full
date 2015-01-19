//=============================================================================
// Fire.fx by Frank Luna (C) 2011 All Rights Reserved.
//
// Fire particle system.  Particles are emitted directly in world space.
//=============================================================================


//***********************************************
// GLOBALS                                      *
//***********************************************

cbuffer cbPerFrame
{
	float3 gEyePosW;
	float3 gEmitPosW;
	float gGameTime;
	float gTimeStep;
	float4x4 gViewProj; 
	float3 g_TotalVel ;
	float g_FrameRate;
	float dirLightIntensity;
	float g_ResponseDirLight;

};
float g_Near; 
float g_Far; 
#define PI 3.14159265
Texture2DArray rainTextureArray;



matrix g_mProjection;
 
 
 float2 g_texcoords[4] = 
    { 
        float2(0,1), 
        float2(1,1),
        float2(0,0),
        float2(1,0),
    };
//normalization factors for the rain textures, one per texture
    float g_rainfactors[370] = 
    {

        0.004535 , 0.014777 , 0.012512 , 0.130630 , 0.013893 , 0.125165 , 0.011809 , 0.244907 , 0.010722 , 0.218252,
        0.011450 , 0.016406 , 0.015855 , 0.055476 , 0.015024 , 0.067772 , 0.021120 , 0.118653 , 0.018705 , 0.142495, 
        0.004249 , 0.017267 , 0.042737 , 0.036384 , 0.043433 , 0.039413 , 0.058746 , 0.038396 , 0.065664 , 0.054761, 
        0.002484 , 0.003707 , 0.004456 , 0.006006 , 0.004805 , 0.006021 , 0.004263 , 0.007299 , 0.004665 , 0.007037, 
        0.002403 , 0.004809 , 0.004978 , 0.005211 , 0.004855 , 0.004936 , 0.006266 , 0.007787 , 0.006973 , 0.007911, 
        0.004843 , 0.007565 , 0.007675 , 0.011109 , 0.007726 , 0.012165 , 0.013179 , 0.021546 , 0.013247 , 0.012964, 
        0.105644 , 0.126661 , 0.128746 , 0.101296 , 0.123779 , 0.106198 , 0.123470 , 0.129170 , 0.116610 , 0.137528, 
        0.302834 , 0.379777 , 0.392745 , 0.339152 , 0.395508 , 0.334227 , 0.374641 , 0.503066 , 0.387906 , 0.519618, 
        0.414521 , 0.521799 , 0.521648 , 0.498219 , 0.511921 , 0.490866 , 0.523137 , 0.713744 , 0.516829 , 0.743649, 
        0.009892 , 0.013868 , 0.034567 , 0.025788 , 0.034729 , 0.036399 , 0.030606 , 0.017303 , 0.051809 , 0.030852, 
        0.018874 , 0.027152 , 0.031625 , 0.023033 , 0.038150 , 0.024483 , 0.029034 , 0.021801 , 0.037730 , 0.016639, 
        0.002868 , 0.004127 , 0.133022 , 0.013847 , 0.123368 , 0.012993 , 0.122183 , 0.015031 , 0.126043 , 0.015916, 
        0.002030 , 0.002807 , 0.065443 , 0.002752 , 0.069440 , 0.002810 , 0.081357 , 0.002721 , 0.076409 , 0.002990, 
        0.002425 , 0.003250 , 0.003180 , 0.011331 , 0.002957 , 0.011551 , 0.003387 , 0.006086 , 0.002928 , 0.005548, 
        0.003664 , 0.004258 , 0.004269 , 0.009404 , 0.003925 , 0.009233 , 0.004224 , 0.009405 , 0.004014 , 0.008435, 
        0.038058 , 0.040362 , 0.035946 , 0.072104 , 0.038315 , 0.078789 , 0.037069 , 0.077795 , 0.042554 , 0.073945, 
        0.124160 , 0.122589 , 0.121798 , 0.201886 , 0.122283 , 0.214549 , 0.118196 , 0.192104 , 0.122268 , 0.209397, 
        0.185212 , 0.181729 , 0.194527 , 0.420721 , 0.191558 , 0.437096 , 0.199995 , 0.373842 , 0.192217 , 0.386263, 
        0.003520 , 0.053502 , 0.060764 , 0.035197 , 0.055078 , 0.036764 , 0.048231 , 0.052671 , 0.050826 , 0.044863, 
        0.002254 , 0.023290 , 0.082858 , 0.043008 , 0.073780 , 0.035838 , 0.080650 , 0.071433 , 0.073493 , 0.026725, 
        0.002181 , 0.002203 , 0.112864 , 0.060140 , 0.115635 , 0.065531 , 0.093277 , 0.094123 , 0.093125 , 0.144290, 
        0.002397 , 0.002369 , 0.043241 , 0.002518 , 0.040455 , 0.002656 , 0.002540 , 0.090915 , 0.002443 , 0.101604, 
        0.002598 , 0.002547 , 0.002748 , 0.002939 , 0.002599 , 0.003395 , 0.002733 , 0.003774 , 0.002659 , 0.004583, 
        0.003277 , 0.003176 , 0.003265 , 0.004301 , 0.003160 , 0.004517 , 0.003833 , 0.008354 , 0.003140 , 0.009214, 
        0.008558 , 0.007646 , 0.007622 , 0.026437 , 0.007633 , 0.021560 , 0.007622 , 0.017570 , 0.007632 , 0.018037, 
        0.031062 , 0.028428 , 0.028428 , 0.108300 , 0.028751 , 0.111013 , 0.028428 , 0.048661 , 0.028699 , 0.061490, 
        0.051063 , 0.047597 , 0.048824 , 0.129541 , 0.045247 , 0.124975 , 0.047804 , 0.128904 , 0.045053 , 0.119087, 
        0.002197 , 0.002552 , 0.002098 , 0.200688 , 0.002073 , 0.102060 , 0.002111 , 0.163116 , 0.002125 , 0.165419, 
        0.002060 , 0.002504 , 0.002105 , 0.166820 , 0.002117 , 0.144274 , 0.005074 , 0.143881 , 0.004875 , 0.205333, 
        0.001852 , 0.002184 , 0.002167 , 0.163804 , 0.002132 , 0.212644 , 0.003431 , 0.244546 , 0.004205 , 0.315848, 
        0.002450 , 0.002360 , 0.002243 , 0.154635 , 0.002246 , 0.148259 , 0.002239 , 0.348694 , 0.002265 , 0.368426, 
        0.002321 , 0.002393 , 0.002376 , 0.074124 , 0.002439 , 0.126918 , 0.002453 , 0.439270 , 0.002416 , 0.489812, 
        0.002484 , 0.002629 , 0.002559 , 0.150246 , 0.002579 , 0.140103 , 0.002548 , 0.493103 , 0.002637 , 0.509481, 
        0.002960 , 0.002952 , 0.002880 , 0.294884 , 0.002758 , 0.332805 , 0.002727 , 0.455842 , 0.002816 , 0.431807, 
        0.003099 , 0.003028 , 0.002927 , 0.387154 , 0.002899 , 0.397946 , 0.002957 , 0.261333 , 0.002909 , 0.148548, 
        0.004887 , 0.004884 , 0.006581 , 0.414647 , 0.003735 , 0.431317 , 0.006426 , 0.148997 , 0.003736 , 0.080715, 
        0.001969 , 0.002159 , 0.002325 , 0.200211 , 0.002288 , 0.202137 , 0.002289 , 0.595331 , 0.002311 , 0.636097 

        };
cbuffer cbFixed
{
	// Net constant acceleration used to accerlate the particles.
	//float3 gAccelW = {0.0f, 17.8f, 0.0f};
	float3 gAccelW = {-10.0f, -119.8f, 20.0f};
	// Texture coordinates used to stretch texture over quad 
	// when we expand point particle into a quad.
	float2 gQuadTexC[4] = 
	{
		float2(0.0f, 1.0f),
		float2(1.0f, 1.0f),
		float2(0.0f, 0.0f),
		float2(1.0f, 0.0f)
	};
};
 bool cullSprite( float3 position, float SpriteSize)
{
    float4 vpos = mul(float4(position,1), gViewProj);
    
    
    if( (vpos.z < (g_Near - SpriteSize )) || ( vpos.z > (g_Far + SpriteSize)) ) 
    {
        return true;
    }
    else 
    {
        float4 ppos = mul( vpos, g_mProjection);
        float wext = ppos.w + SpriteSize;
        if( (ppos.x < -wext) || (ppos.x > wext) ||
            (ppos.y < -wext) || (ppos.y > wext) ) {
            return true;
        }
        else 
        {
            return false;
        }
    }
    
    return false;
}
// Random texture used to generate random numbers in shaders.
Texture1D gRandomTex;
 
SamplerState samLinear
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = WRAP;
	AddressV = WRAP;
};
SamplerState samAniso
{
    Filter = ANISOTROPIC;
    AddressU = WRAP;
    AddressV = WRAP;
}; 
DepthStencilState DisableDepth
{
    DepthEnable = FALSE;
    DepthWriteMask = ZERO;
};

DepthStencilState NoDepthWrites
{
    DepthEnable = TRUE;
    DepthWriteMask = ZERO;
};

BlendState AdditiveBlending
{
    AlphaToCoverageEnable = FALSE;
    BlendEnable[0] = TRUE;
    SrcBlend = SRC_ALPHA;
    DestBlend = ONE;
    BlendOp = ADD;
    SrcBlendAlpha = ZERO;
    DestBlendAlpha = ZERO;
    BlendOpAlpha = ADD;
    RenderTargetWriteMask[0] = 0x0F;
};

 
//***********************************************
// STREAM-OUT TECH                              *
//***********************************************

#define PT_EMITTER 0
#define PT_FLARE 1
 
struct Particle
{
	float3 pos : POSITION;
	float3 seed : SEED;
	float3 speed : SPEED;
	float random : RAND;
	uint Type : TYPE;	
};
struct PSSceneIn
{
    float4 pos : SV_Position;
    float3 lightDir   : LIGHT;
    float3 pointLightDir : LIGHT2;
    float3 eyeVec  : EYE;
    float2 tex : TEXTURE0;
    uint type  : TYPE;
    float random : RAND;
};

 
[maxvertexcount(1)]
void StreamOutGS( point Particle gin[1], inout PointStream<Particle> SpriteStream)
{	
	gin[0].pos += float3(2,-5,2)/g_FrameRate/* + g_TotalVel*/;//未生效

	if(gin[0].pos.y <=gEyePosW.y+35*gin[0].seed.y-14)
	{                                                                                                                                                                                                          
	     gin[0].pos = gEyePosW.xyz + 35*gin[0].seed ;	
	}
		SpriteStream.Append(gin[0]);	

}


Particle StreamOutVS(Particle vin)
{
	return vin;
}
GeometryShader gsStreamOut = ConstructGSWithSO( 
	CompileShader( gs_5_0, StreamOutGS() ), 
	"POSITION.xyz; SEED.xyz; RAND.x; SPEED.xyz; TYPE.x" );
	
technique11 StreamOutTech
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_5_0, StreamOutVS() ) );
        SetGeometryShader( gsStreamOut );
        SetPixelShader(NULL);
        
        // we must also disable the depth buffer for stream-out only
        SetDepthStencilState( DisableDepth, 0 );
    }
}

//***********************************************
// DRAW TECH                                    *
//***********************************************



struct GeoOut
{
	float4 PosH  : SV_Position;
	float2 Tex   : TEXTURE0;
};
void rainResponse(PSSceneIn input, float3 lightVector, float lightIntensity, float3 lightColor, float3 eyeVector, bool fallOffFactor, inout float4 rainResponseVal)
{
    
    float opacity = 0.0;

    float fallOff;
    if(fallOffFactor)
    {  
        float distToLight = length(lightVector);
        fallOff = 1.0/( distToLight * distToLight);
        fallOff = saturate(fallOff);   
    }
    else
    {  fallOff = 1;
    }

    if(fallOff > 0.01 && lightIntensity > 0.01 )
    {
        float3 dropDir = g_TotalVel;

        #define MAX_VIDX 4
        #define MAX_HIDX 8
        // Inputs: lightVector, eyeVector, dropDir
        float3 L = normalize(lightVector);
        float3 E = normalize(eyeVector);
        float3 N = normalize(dropDir);
        
        bool is_EpLp_angle_ccw = true;
        float hangle = 0;
        float vangle = abs( (acos(dot(L,N)) * 180/PI) - 90 ); // 0 to 90
        
        {
            float3 Lp = normalize( L - dot(L,N)*N );
            float3 Ep = normalize( E - dot(E,N)*N );
            hangle = acos( dot(Ep,Lp) ) * 180/PI;  // 0 to 180
            hangle = (hangle-10)/20.0;           // -0.5 to 8.5
            is_EpLp_angle_ccw = dot( N, cross(Ep,Lp)) > 0;
        }
        
        if(vangle>=88.0)
        {
            hangle = 0;
            is_EpLp_angle_ccw = true;
        }
                
        vangle = (vangle-10.0)/20.0; // -0.5 to 4.5
        
        // Outputs:
        // verticalLightIndex[1|2] - two indices in the vertical direction
        // t - fraction at which the vangle is between these two indices (for lerp)
        int verticalLightIndex1 = floor(vangle); // 0 to 5
        int verticalLightIndex2 = min(MAX_VIDX, (verticalLightIndex1 + 1) );
        verticalLightIndex1 = max(0, verticalLightIndex1);
        float t = frac(vangle);

        // textureCoordsH[1|2] used in case we need to flip the texture horizontally
        float textureCoordsH1 = input.tex.x;
        float textureCoordsH2 = input.tex.x;
        
        // horizontalLightIndex[1|2] - two indices in the horizontal direction
        // s - fraction at which the hangle is between these two indices (for lerp)
        int horizontalLightIndex1 = 0;
        int horizontalLightIndex2 = 0;
        float s = 0;
        
        s = frac(hangle);
        horizontalLightIndex1 = floor(hangle); // 0 to 8
        horizontalLightIndex2 = horizontalLightIndex1+1;
        if( horizontalLightIndex1 < 0 )
        {
            horizontalLightIndex1 = 0;
            horizontalLightIndex2 = 0;
        }
                   
        if( is_EpLp_angle_ccw )
        {
            if( horizontalLightIndex2 > MAX_HIDX ) 
            {
                horizontalLightIndex2 = MAX_HIDX;
                textureCoordsH2 = 1.0 - textureCoordsH2;
            }
        }
        else
        {
            textureCoordsH1 = 1.0 - textureCoordsH1;
            if( horizontalLightIndex2 > MAX_HIDX ) 
            {
                horizontalLightIndex2 = MAX_HIDX;
            } else 
            {
                textureCoordsH2 = 1.0 - textureCoordsH2;
            }
        }
                
        if( verticalLightIndex1 >= MAX_VIDX )
        {
            textureCoordsH2 = input.tex.x;
            horizontalLightIndex1 = 0;
            horizontalLightIndex2 = 0;
            s = 0;
        }
        
        // Generate the final texture coordinates for each sample
        uint type = input.type;
        uint2 texIndicesV1 = uint2(verticalLightIndex1*90 + horizontalLightIndex1*10 + type,
                                     verticalLightIndex1*90 + horizontalLightIndex2*10 + type);
        float3 tex1 = float3(textureCoordsH1, input.tex.y, texIndicesV1.x);
        float3 tex2 = float3(textureCoordsH2, input.tex.y, texIndicesV1.y);
        if( (verticalLightIndex1<4) && (verticalLightIndex2>=4) ) 
        {
            s = 0;
            horizontalLightIndex1 = 0;
            horizontalLightIndex2 = 0;
            textureCoordsH1 = input.tex.x;
            textureCoordsH2 = input.tex.x;
        }
        
        uint2 texIndicesV2 = uint2(verticalLightIndex2*90 + horizontalLightIndex1*10 + type,
                                     verticalLightIndex2*90 + horizontalLightIndex2*10 + type);
        float3 tex3 = float3(textureCoordsH1, input.tex.y, texIndicesV2.x);        
        float3 tex4 = float3(textureCoordsH2, input.tex.y, texIndicesV2.y);

        // Sample opacity from the textures
        float col1 = rainTextureArray.Sample( samAniso, tex1).x * g_rainfactors[texIndicesV1.x];//纹理数组采样
        float col2 = rainTextureArray.Sample( samAniso, tex2).x * g_rainfactors[texIndicesV1.y];//不知道为什么这边要加.x
        float col3 = rainTextureArray.Sample( samAniso, tex3).x * g_rainfactors[texIndicesV2.x];
        float col4 = rainTextureArray.Sample( samAniso, tex4).x * g_rainfactors[texIndicesV2.y];

        // Compute interpolated opacity using the s and t factors
        float hOpacity1 = lerp(col1,col2,s);
        float hOpacity2 = lerp(col3,col4,s);
        opacity = lerp(hOpacity1,hOpacity2,t);
        opacity = pow(opacity,0.7); // inverse gamma correction (expand dynamic range)
        opacity = 4*lightIntensity * opacity * fallOff;
    }
         
   rainResponseVal = float4(lightColor,opacity);

}
void GenRainSpriteVertices(float3 worldPos, float3 velVec, float3 eyePos, out float3 outPos[4])
{
    float height = 1.0/2.0;
    float width = height/10.0;

    velVec = normalize(velVec);
    float3 eyeVec = eyePos - worldPos;
    float3 eyeOnVelVecPlane = eyePos - ((dot(eyeVec, velVec)) * velVec);
    float3 projectedEyeVec = eyeOnVelVecPlane - worldPos;
    float3 sideVec = normalize(cross(projectedEyeVec, velVec));
    
    outPos[0] =  worldPos - (sideVec * 0.5*width);
    outPos[1] = outPos[0] + (sideVec * width);
    outPos[2] = outPos[0] + (velVec * height);
    outPos[3] = outPos[2] + (sideVec * width );
}
[maxvertexcount(4)]
void GSRenderRainCheap(point Particle input[1], inout TriangleStream<GeoOut> SpriteStream)
{
   
    GeoOut output = (GeoOut)0;
    float3 pos[4];
    GenRainSpriteVertices(input[0].pos, float3(2,-5,2)/g_FrameRate/*+ g_TotalVel*/, gEyePosW, pos);
     
    output.PosH = mul( float4(pos[0].x,pos[0].z,pos[0].y,1.0), gViewProj );
    output.Tex = float2(0,1);
    SpriteStream.Append(output);
        
    output.PosH = mul( float4(pos[1].x,pos[1].z,pos[1].y,1.0), gViewProj );
    output.Tex = float2(1,1);
    SpriteStream.Append(output);
        
    output.PosH = mul( float4(pos[2].x,pos[2].z,pos[2].y,1.0), gViewProj );
    output.Tex = float2(0,0);
    SpriteStream.Append(output);
                
    output.PosH = mul( float4(pos[3].x,pos[3].z,pos[3].y,1.0), gViewProj );
    output.Tex = float2(1,0);
    SpriteStream.Append(output);
        
    SpriteStream.RestartStrip();
       
}

// The draw GS just expands points into camera facing quads.
[maxvertexcount(4)]
void DrawGS(point Particle gin[1], inout TriangleStream<GeoOut> triStream)
			
{	
	// do not draw emitter particles.

		//
		// Compute world matrix so that billboard faces the camera.
		//
		float3 look  = normalize(gEyePosW.xyz - gin[0].pos);
		float3 right = normalize(cross(float3(0,1,0), look));
		float3 up    = cross(look, right);
		
		//
		// Compute triangle strip vertices (quad) in world space.
		//
		float spritesize = 1.0;
		float halfWidth  = 0.05f*spritesize;
		float halfHeight = 0.5f*spritesize;
	
		float4 v[4];

		v[0] = float4(gin[0].pos + halfWidth*right - halfHeight*up, 1.0f);
		v[1] = float4(gin[0].pos + halfWidth*right + halfHeight*up, 1.0f);
		v[2] = float4(gin[0].pos - halfWidth*right - halfHeight*up, 1.0f);
		v[3] = float4(gin[0].pos - halfWidth*right + halfHeight*up, 1.0f);
		
		//
		// Transform quad vertices to world space and output 
		// them as a triangle strip.
		//
		GeoOut gout;
		[unroll]
		for(int i = 0; i < 4; ++i)
		{
			gout.PosH  = mul(float4(v[i].x,v[i].z,v[i].y,1.0f), gViewProj);//本框架y,z是相反的
			gout.Tex   = gQuadTexC[i];
		
			triStream.Append(gout);
		}	
		triStream.RestartStrip();
}

float4 DrawPS(GeoOut pin) : SV_TARGET
{
	//float dirLightIntensity = 0.27;
	//float g_ResponseDirLight = 0.9;
	return float4(1,1,1,0.1);
}
RasterizerState CullNone
{
    MultiSampleEnable = False;
    CullMode=None;
};
technique11 DrawTech
{
    pass P0
    {
		SetVertexShader(   CompileShader( vs_5_0, StreamOutVS() ) );
		//SetGeometryShader( CompileShader( gs_5_0, GSRenderRainCheap() ) );
		SetGeometryShader( CompileShader( gs_5_0, DrawGS() ) );
        SetPixelShader(    CompileShader( ps_5_0, DrawPS() ) );
        
        SetBlendState(AdditiveBlending, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff);
        SetDepthStencilState( NoDepthWrites, 0 );
		SetRasterizerState( CullNone );
    }
}

// GS for rendering rain as point sprites.  Takes a point and turns it into 2 tris.
[maxvertexcount(4)]
void GSRenderRain(point Particle input[1], inout TriangleStream<PSSceneIn> SpriteStream)
{
	float dirLightIntensity = 1.0;
	float g_PointLightIntensity = 2.0;
	float g_ResponseDirLight = 1.0;
	float g_ResponsePointLight = 1.0;
    float totalIntensity = g_PointLightIntensity*g_ResponsePointLight + dirLightIntensity*g_ResponseDirLight;//光的总强度
    if(!cullSprite(input[0].pos,2*1.0) && totalIntensity > 0)
    {    
        PSSceneIn output = (PSSceneIn)0;
        output.type = input[0].Type;
        output.random = input[0].random;
       
        float3 pos[4];
        GenRainSpriteVertices(input[0].pos.xyz, input[0].speed.xyz/g_FrameRate + g_TotalVel, gEyePosW, pos);
        
		//选出离它近的点光源
		 float3 g_PointLightPos = float3(  3.7,5.8,3.15);     
		 float3 g_PointLightPos2 = float3(-3.7,5.8,3.15);

        float3 closestPointLight = g_PointLightPos;
        float closestDistance = length(g_PointLightPos - pos[0]);
        if( length(g_PointLightPos2 - pos[0]) < closestDistance )
           closestPointLight = g_PointLightPos2;

        float3 g_lightPos = float3(10,10,0);

        output.pos = mul( float4(pos[0].x,pos[0].z,pos[0].y,1.0), gViewProj );
        output.lightDir = g_lightPos - pos[0];
        output.pointLightDir = closestPointLight - pos[0];
        output.eyeVec = gEyePosW - pos[0];
        output.tex = g_texcoords[0];
        SpriteStream.Append(output);
                
        output.pos = mul( float4(pos[1].x,pos[1].z,pos[1].y,1.0), gViewProj );
        output.lightDir = g_lightPos - pos[1];
        output.pointLightDir = closestPointLight - pos[1];
        output.eyeVec = gEyePosW - pos[1];
        output.tex = g_texcoords[1];
        SpriteStream.Append(output);
        
        output.pos = mul( float4(pos[2].x,pos[2].z,pos[2].y,1.0), gViewProj );
        output.lightDir = g_lightPos - pos[2];
        output.pointLightDir = closestPointLight - pos[2];
        output.eyeVec = gEyePosW - pos[2];
        output.tex = g_texcoords[2];
        SpriteStream.Append(output);
                
        output.pos = mul( float4(pos[3].x,pos[3].z,pos[3].y,1.0), gViewProj );
        output.lightDir = g_lightPos - pos[3];
        output.pointLightDir = closestPointLight - pos[3];
        output.eyeVec = gEyePosW - pos[3];
        output.tex = g_texcoords[3];
        SpriteStream.Append(output);
        
        SpriteStream.RestartStrip();
    }   
}

float4 PSRenderRain(PSSceneIn input) : SV_Target
{     
      //return float4(1,0,0,0.1);
       
      //directional lighting---------------------------------------------------------------------------------
      float4 directionalLight;
    //  rainResponse(input, input.lightDir, 2.0*dirLightIntensity*g_ResponseDirLight*input.random, float3(1.0,1.0,1.0), input.eyeVec, false, directionalLight);
	  directionalLight = float4(1.0f,1.0f,1.0f,1.0f);

      //point lighting---------------------------------------------------------------------------------------
      float4 pointLight = float4(0,0,0,0);
      float3 g_SpotLightDir = float3(0,-1,0);
	  bool g_useSpotLight = true;
	  float g_cosSpotlightAngle = 0.8;
      float3 L = normalize( input.pointLightDir );
      float angleToSpotLight = dot(-L, g_SpotLightDir);
      float4 pointLightColor = float4(1.0,1.0,1.0,1.0);

      if( !g_useSpotLight || g_useSpotLight && angleToSpotLight > g_cosSpotlightAngle )
         // rainResponse(input, input.pointLightDir, 2*g_PointLightIntensity*g_ResponsePointLight*input.random, pointLightColor.xyz, input.eyeVec, true,pointLight);
			pointLight = float4(pointLightColor.xyz,1.0f);
      float totalOpacity = pointLight.a+directionalLight.a;
      return float4( float3(pointLight.rgb*pointLight.a/totalOpacity + directionalLight.rgb*directionalLight.a/totalOpacity), totalOpacity);//取点光源*百分比+平行光*百分比（颜色）
}

technique11 DrawRain
{
    pass P0
    {
		SetVertexShader(   CompileShader( vs_5_0, StreamOutVS() ) );
		SetGeometryShader( CompileShader( gs_5_0, GSRenderRain() ) );
        SetPixelShader(    CompileShader( ps_5_0, PSRenderRain() ) );
        
        SetBlendState(AdditiveBlending, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xffffffff);
        SetDepthStencilState( NoDepthWrites, 0 );
	//	SetDepthStencilState( DisableDepth, 0 );
		SetRasterizerState( CullNone );
    }
}