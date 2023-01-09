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

const clientId = "becbabeb5d0d4204b5b99689751e71ef";

const viewerOptions = {
    // embedMode: "IN_LINE",
    embedMode: "LIGHT_BOX",
    // embedMode: "SIZED_CONTAINER",
    // embedMode: "FULL_WINDOW",
    defaultViewMode: "FIT_PAGE",
    // showDownloadPDF: false,
    // showPrintPDF: false,
    enableFormFilling: false,
    // showZoomControl: false,
    showThumbnails: false,
    showBookmarks: false,
    showAnnotationTools: false,
    showFullScreen: true,
    // enableLinearization: true,
};

function fetchPDF(urlToPDF) {
    return new Promise((resolve) => {
        fetch(urlToPDF)
            .then((resolve) => resolve.blob())
            .then((blob) => {
                resolve(blob.arrayBuffer());
            })
    })
}

function showPDF(urlToPDF) {
    var adobeDCView = new AdobeDC.View({
            clientId: clientId
        });
        var previewFilePromise = adobeDCView.previewFile(
            {
                content: { promise: fetchPDF(urlToPDF) },
                metaData: { fileName: urlToPDF.split("/").slice(-1)[0] }
            },
            viewerOptions
        );
}

document.addEventListener("adobe_dc_view_sdk.ready", function () {

    document.getElementById("showPDF-int").addEventListener("click", function () {
        showPDF("https://arxiv.org/pdf/2009.07385.pdf")
    });

    document.getElementById("showPDF-gpr").addEventListener("click", function () {
        showPDF("https://arxiv.org/pdf/2206.09976.pdf")
    });
});

// Add arrayBuffer if necessary i.e. Safari
(function () {
    if (Blob.arrayBuffer != "function") {
        Blob.prototype.arrayBuffer = myArrayBuffer;
    }

    function myArrayBuffer() {
        return new Promise((resolve) => {
            let fileReader = new FileReader();
            fileReader.onload = () => {
                resolve(fileReader.result);
            };
            fileReader.readAsArrayBuffer(this);
        });
    }
})();


// document.addEventListener("adobe_dc_view_sdk.ready", function()
// {
//     var adobeDCView = new AdobeDC.View({clientId: "becbabeb5d0d4204b5b99689751e71ef", divId: "adobe-dc-view"});
//     adobeDCView.previewFile(
//         {
//             content:{location: {url: "https://arxiv.org/pdf/2009.07385.pdf"}},
//             // content:{location: {promise: filePromise}},
//             // metaData:{fileName: "Bodea Brochure.pdf"}
//             metaData:{fileName: ""}
//         },
//         {
//             // embedMode: "IN_LINE",
//             // embedMode: "LIGHT_BOX",
//             // embedMode: "SIZED_CONTAINER",
//             embedMode: "FULL_WINDOW",
//             showDownloadPDF: false,
//             showPrintPDF: false,
//             enableFormFilling: false,
//             showZoomControl: false,
//             showThumbnails: false,
//             showBookmarks: false,
//             showAnnotationTools: false,
//             showFullScreen: true,
//             enableLinearization: true,
//         }
//     );
// });

// Change the link of logo to index.html
document.addEventListener("DOMContentLoaded", function() {

    // Find an "<a>" element whose href ends with "contents.html"
    const match = document.querySelector("a[href*='contents.html']");

    // Replace "contents.html" with "index.html"
    if (match){
        match.href = match.href.replace('contents.html', 'index.html')
    }
});
