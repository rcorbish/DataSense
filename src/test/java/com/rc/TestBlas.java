package com.rc;

import org.junit.BeforeClass;

public class TestBlas extends TestBase {

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		System.getProperties().setProperty("compute_library",  "openblas" ) ;
	}

}
