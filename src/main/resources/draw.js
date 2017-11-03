
function drawCorrelations( ctx, R, w, h ) {

	var dx = w / R.length
	var dy = h / R.length

	for( var i=0 ; i<R.length ; i++ ) {
		for( var j=0 ; j<R.length ; j++ ) {
			if( i==j ) continue ;
			ctx.beginPath()
			const r = Math.round( Math.abs(R[i][j]) * 0xf).toString(16) 
			if( R[i][j]<0 ) {
				ctx.fillStyle = "#" +  r + "00" 
			} else {
				ctx.fillStyle = "#0" + r + "0" 
			}  
			ctx.fillRect( i*dx, j*dy, dx, dy ) 
			ctx.fill()
		} 
	} 
}


function drawLinear( ctx, Y, YH, w, h ) {

	var dx = w / Y.M
	//var dy = h / R.length
	
	ctx.beginPath()
	ctx.strokeStyle = "#fff"
			
	var x = 0 
	var y = Y.data[0]
	
	for( var i=1 ; i<Y.M ; i++ ) {
		x += dx
		y = h - Y.data[i] * 25 - 10 
		ctx.lineTo( x, y )
	}
	ctx.stroke()
	
	ctx.beginPath()
	ctx.strokeStyle = "#0f0"
			
	x = 0 
	y = YH.data[0]
	
	for( var i=1 ; i<YH.M ; i++ ) {
		x += dx
		y = h - YH.data[i] * 25 - 10 
		ctx.lineTo( x, y )
	}
	ctx.stroke()

}
