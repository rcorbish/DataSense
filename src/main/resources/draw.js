
function drawCorrelations( ctx, R, w, h ) {

	var dx = w / R.length
	var dy = h / R.length

	for( var i=0 ; i<R.length ; i++ ) {
		for( var j=i+1 ; j<R.length ; j++ ) {
			if( i==j ) continue ;
			ctx.beginPath()
			const r = Math.round( Math.abs(R[i][j]) * 0xf).toString(16) 
			if( R[i][j]<0 ) {
				ctx.fillStyle = "#" +  r + "00" 
			} else {
				ctx.fillStyle = "#0" + r + "0" 
			}  
			ctx.fillRect( j*dx, i*dy, dx, dy ) 
			ctx.fill()
		} 
	} 
}



function drawStatistics( ctx, obj, w, h ) {

		const stats = [ obj.minimum, obj.mean, obj.median, 
						obj.maximum, obj.stddev, obj.skewness, 
						obj.kurtosis, obj.countDistinct ]
	
		const N=stats[0].M 

		var dx = w / N
		var dy = h / stats.length
	

		for( var i=0 ; i<stats.length ; i++ ) {
			var data = normalize( stats[i].data )
			for( var j=0 ; j<N ; j++ ) {
				ctx.beginPath()
				const r = Math.round( Math.abs(data[j]) * 0xf).toString(16) 
				if( data[j]<0 ) {
					ctx.fillStyle = "#" +  r + "00" 
				} else {
					ctx.fillStyle = "#0" + r + "0" 
				}  
				ctx.fillRect( j*dx, i*dy, dx, dy ) 
				ctx.fill()
			} 
		} 
	}
	
function normalize( data ) {
	var min = Math.min.apply( null, data ) 
	var max = Math.max.apply( null, data ) 

	rc = data.map(d => ( 2 * (d - min) / (max-min) - 1.0 ) ) 

	return rc ;
}

function drawLinear( ctx, Y, YH, w, h ) {

	var dx = w / Y.M
	//var dy = h / R.length
	
	var min = Math.min.apply( null, Y.data ) 
	var max = Math.max.apply( null, Y.data ) 
	var range = ( h - 50 ) / ( max - min ) 
	

	ctx.beginPath()
	ctx.strokeStyle = "#fff"

	var x = 0 
	var y = h - ( Y.data[0] - min ) * range - 25
	ctx.moveTo( x, y ) 

	for( var i=1 ; i<Y.M ; i++ ) {
		x += dx
		y = h - ( Y.data[i] - min ) * range - 25   
		ctx.lineTo( x, y )
	}
	ctx.stroke()
	
	ctx.beginPath()
	ctx.strokeStyle = "#0f0"
			
	x = 0 
	y = h - ( YH.data[0] - min ) * range - 25
	ctx.moveTo( x, y ) 
	for( var i=1 ; i<YH.M ; i++ ) {
		x += dx
		y = h - ( YH.data[i] - min ) * range - 25 
		ctx.lineTo( x, y )
	}
	ctx.stroke()
}


function drawLogistic( ctx, yHist, yhHist, ymHist, w, h ) {

	const dx = (w-50) / yHist.length
	const barWidth = dx / 2 - 5
	var max = Math.max.apply( null, yHist ) 
	max = Math.max.apply( max, yhHist ) 
	const range = ( h - 50 ) / max  
	
	ctx.beginPath()
	ctx.strokeStyle = "#666"
	ctx.fillStyle = "#666"
			
	var x = 25
	var y 
	for( var i=0 ; i<yHist.length ; i++ ) {
		//ctx.moveTo( x, h ) 
		y = h - yHist[i] * range    
		//ctx.lineTo( x, y )
		ctx.fillRect( x, y-25, barWidth, h-y )
		x += dx
	}
	ctx.stroke()

	ctx.beginPath()
	ctx.strokeStyle = "#999"
	ctx.fillStyle = "#999"
			
	x = 25
	for( var i=0 ; i<ymHist.length ; i++ ) {
		//ctx.moveTo( x, h ) 
		y = h - ymHist[i] * range    
		//ctx.lineTo( x, y )
		ctx.fillRect( x, y-25, barWidth, h-y )
		x += dx
	}
	ctx.stroke()

	ctx.beginPath()
	ctx.strokeStyle = "#060"
	ctx.fillStyle = "#060"
			
	x = 25+barWidth
	for( var i=0 ; i<yhHist.length ; i++ ) {
//		ctx.moveTo( x, h ) 
		y = h - yhHist[i] * range 
//		ctx.lineTo( x, y )
		ctx.fillRect( x, y-25, barWidth, h-y )
		x += dx
	}
	ctx.stroke()

	ctx.beginPath()
	ctx.strokeStyle = "#090"
	ctx.fillStyle = "#090"
			
	x = 25+barWidth
	for( var i=0 ; i<ymHist.length ; i++ ) {
//		ctx.moveTo( x, h ) 
		y = h - ymHist[i] * range 
//		ctx.lineTo( x, y )
		ctx.fillRect( x, y-25, barWidth, h-y )
		x += dx
	}
	ctx.stroke()
}
