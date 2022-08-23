// window.addEventListener("load", function(event)
// {
//     var images = document.getElementsByTagName("img");
//     for(var i=0; i < images.length; i++) {
//         // images[i].src = "someImage.jpg";
//         // console.log(images[i].src);
//     }
// });

// How to embed PDF files in sphinx using Adobe Embed PDF API:
//
// 1. In /docs/source/_templates/layout.html add this to the scripts:
// <script src="https://documentcloud.adobe.com/view-sdk/viewer.js"></script>
//
// 2. In the sphinx rts file, create a div by the following raw html directive.
//    Make sure the div id is the same.
//
// .. raw:: html
//
//     <div id="adobe-dc-view" style="width: 800px;"></div>
//
// 3. Then uncomment the following function here, and add the url of the PDF file.
//
// document.addEventListener("adobe_dc_view_sdk.ready", function()
// {
//     var adobeDCView = new AdobeDC.View({clientId: "becbabeb5d0d4204b5b99689751e71ef", divId: "adobe-dc-view"});
//     adobeDCView.previewFile(
//         {
//             content:{location: {url: "https://documentcloud.adobe.com/view-sdk-demo/PDFs/Bodea Brochure.pdf"}},
//             // content:{location: {promise: filePromise}},
//             metaData:{fileName: "Bodea Brochure.pdf"}
//         },
//         {
//             embedMode: "IN_LINE",
//             showDownloadPDF: false,
//             showPrintPDF: false,
//             showZoomControl: false,
//             showThumbnails: false,
//             showBookmarks: false,
//             showAnnotationTools: false,
//         }
//     );
// });
