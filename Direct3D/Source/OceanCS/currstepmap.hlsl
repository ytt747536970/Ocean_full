struct VS_QUAD_OUTPUT
{
    float4 Position		: SV_POSITION;	// vertex position
    float2 TexCoord		: TEXCOORD0;	// vertex texture coords 
};

VS_QUAD_OUTPUT QuadVS(float4 vPos : POSITION)
{
	VS_QUAD_OUTPUT Output;

	Output.Position = vPos;
	Output.TexCoord.x = 0.5f + vPos.x * 0.5f;
	Output.TexCoord.y = 0.5f - vPos.y * 0.5f;

	return Output;
}
//---------------------------------------- Pixel Shaders ------------------------------------------
float4 CurrStepMapPS(VS_QUAD_OUTPUT In) : SV_Target
{
	VS_OUTPUT Output;
	float2 vPos;
	//不确定是否对？
	vPos.x = (uint)(In.TexCoord.x * (float)g_OutWidth);
	vPos.y = (uint)(In.TexCoord.y * (float)g_OutHeight);

	// Local position
	float4 pos_local = mul(float4(vPos, 0, 1), g_matLocal);
	// UV
	float2 uv_local = pos_local.xy * g_UVScale + g_UVOffset;

	// Blend displacement to avoid tiling artifact
	float3 eye_vec = pos_local.xyz - g_LocalEye;
	float dist_2d = length(eye_vec.xy);
	float blend_factor = (PATCH_BLEND_END - dist_2d) / (PATCH_BLEND_END - PATCH_BLEND_BEGIN);
	blend_factor = clamp(blend_factor, 0, 1);

	// Add perlin noise to distant patches
	float perlin = 0;
	if (blend_factor < 1)
	{
		float2 perlin_tc = uv_local * g_PerlinSize + g_UVBase;
		float perlin_0 = g_texPerlin.SampleLevel(g_samplerPerlin, perlin_tc * g_PerlinOctave.x + g_PerlinMovement, 0).w;
		float perlin_1 = g_texPerlin.SampleLevel(g_samplerPerlin, perlin_tc * g_PerlinOctave.y + g_PerlinMovement, 0).w;
		float perlin_2 = g_texPerlin.SampleLevel(g_samplerPerlin, perlin_tc * g_PerlinOctave.z + g_PerlinMovement, 0).w;
		
		perlin = perlin_0 * g_PerlinAmplitude.x + perlin_1 * g_PerlinAmplitude.y + perlin_2 * g_PerlinAmplitude.z;
	}

	// Displacement map
	float3 displacement = 0;
	if (blend_factor > 0)
		displacement = g_texDisplacement.SampleLevel(g_samplerDisplacement, uv_local, 0).xyz;
	displacement = lerp(float3(0, 0, perlin), displacement, blend_factor);
	//pos_local.xyz += displacement;改动，只偏移y方向的
	pos_local.y += displacement.y;
	pos_local.x += displacement.x;
	pos_local.z += displacement.z;
	// Transform
//	Output.Position = mul(pos_local, g_matWorldViewProj);
//	Output.LocalPos = pos_local.xyz;
	
	// Pass thru texture coordinate
//	Output.TexCoord = uv_local;

	return float4(pos_local.x,pos_local.y,pos_local.z,1); 
}