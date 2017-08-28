#include "Renderer.h"
#include "glinc.h"

/**
 * <!--  renderPre():  -->
 */
void Renderer::renderPre() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
